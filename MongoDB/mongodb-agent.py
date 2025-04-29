import pymongo
import requests
import json
import re
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.markdown import Markdown

class MongoDBAgent:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="test"):
        """Initialize MongoDB connection and Ollama settings"""
        self.console = Console()
        
        with self.console.status("[bold green]Connecting to MongoDB..."):
            self.client = pymongo.MongoClient(mongo_uri)
            self.db = self.client[db_name]
        
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "gemma3:12b"  # Default model, can be changed
        
    def set_database(self, db_name):
        """Change the current database"""
        with self.console.status(f"[bold yellow]Switching to database {db_name}..."):
            self.db = self.client[db_name]
        return f"Database switched to [bold green]{db_name}[/]"
        
    def list_collections(self):
        """List all collections in current database"""
        with self.console.status("[bold green]Fetching collections..."):
            return list(self.db.list_collection_names())
    
    def query_ollama(self, prompt):
        """Send a query to Ollama and get a response"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        with self.console.status(f"[bold cyan]Querying {self.model}..."):
            try:
                response = requests.post(self.ollama_url, headers=headers, json=data)
                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    return f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                return f"Error communicating with Ollama: {str(e)}"
    
    def extract_query_parameters(self, collection_name, user_query):
        """Use Ollama to interpret the user's query and extract MongoDB query parameters"""
        collections = self.list_collections()
        if collection_name not in collections:
            # Try to find the closest matching collection or use the first available one
            if collections:
                suggested_collection = collections[0]
                return {"suggested_collection": suggested_collection, 
                        "original_query": user_query,
                        "error": f"Collection '{collection_name}' does not exist. Available collections: {collections}"}
            else:
                return {"error": f"Collection '{collection_name}' does not exist and no collections are available in the database"}
        
        # Get sample document to understand schema
        with self.console.status(f"[bold blue]Analyzing collection schema..."):
            sample = list(self.db[collection_name].find().limit(1))
            sample_keys = []
            if sample:
                sample_keys = list(sample[0].keys())
        
        system_prompt = f"""
        I want you to act as a MongoDB query generator.
        The collection is: {collection_name}
        The fields in this collection are: {sample_keys}
        
        Based on this user query: "{user_query}"
        Give me ONLY a valid JSON object with the following structure:
        {{
            "filter": {{MongoDB filter conditions}},
            "limit": number,
            "sort": [["field", 1 or -1]],
            "project": {{fields to include or exclude}}
        }}
        
        Don't include explanations, just the JSON object.
        """
        
        llm_response = self.query_ollama(system_prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'({.*})', llm_response.replace('\n', ' '), re.DOTALL)
        if json_match:
            try:
                query_params = json.loads(json_match.group(1))
                return query_params
            except json.JSONDecodeError:
                return {"error": "Failed to parse LLM response as JSON"}
        else:
            return {"error": "No valid JSON found in LLM response"}
    
    def execute_natural_language_query(self, collection_name, user_query):
        """Execute a query based on natural language input"""
        query_params = self.extract_query_parameters(collection_name, user_query)
        
        if "error" in query_params:
            # If there's a suggested alternative collection, try that
            if "suggested_collection" in query_params:
                self.console.print(f"[yellow]Collection '{collection_name}' not found. Trying with '{query_params['suggested_collection']}' instead.[/]")
                return self.execute_natural_language_query(query_params["suggested_collection"], user_query)
            return query_params["error"]
        
        try:
            collection = self.db[collection_name]
            
            filter_cond = query_params.get("filter", {})
            limit = query_params.get("limit", 10)
            sort = query_params.get("sort", None)
            project = query_params.get("project", None)
            
            # Show the generated query for transparency
            self.console.print(Panel(
                Syntax(json.dumps(query_params, indent=2), "json", theme="monokai"),
                title="[bold]Generated MongoDB Query",
                border_style="blue"
            ))
            
            # Start building the query
            with self.console.status("[bold green]Executing MongoDB query..."):
                # Apply projection directly in the find method if it exists
                if project:
                    cursor = collection.find(filter_cond, project)
                else:
                    cursor = collection.find(filter_cond)
                
                # Apply sort if provided
                if sort:
                    try:
                        cursor = cursor.sort(sort)
                    except Exception as e:
                        self.console.print(f"[yellow]Warning: Could not apply sort {sort}: {str(e)}[/]")
                    
                # Apply limit if provided
                if limit:
                    cursor = cursor.limit(limit)
                    
                # Execute and return results
                results = list(cursor)
                return results
            
        except Exception as e:
            return f"Error executing MongoDB query: {str(e)}"
    
    def process_user_input(self, user_input):
        """Process natural language input and determine what action to take"""
        # Get available collections first for context
        available_collections = self.list_collections()
        
        # Ask Ollama to determine what the user wants to do with context of available collections
        system_prompt = f"""
        Based on this user input: "{user_input}"
        The available collections in the database are: {available_collections}
        
        Determine what MongoDB operation they want to perform.
        Respond with ONLY a JSON object that includes:
        - "action": One of ["query", "list_collections", "switch_database"]
        - "collection_name": Choose from one of the available collections for queries: {available_collections}
        - "db_name": The database name (for switch_database)
        
        Just output the JSON, nothing else.
        """
        
        llm_response = self.query_ollama(system_prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'({.*})', llm_response.replace('\n', ' '), re.DOTALL)
        if json_match:
            try:
                intent = json.loads(json_match.group(1))
                
                # Handle different actions
                if intent.get("action") == "list_collections":
                    return {"result": self.list_collections(), "action": "list_collections"}
                
                elif intent.get("action") == "switch_database":
                    db_name = intent.get("db_name")
                    if db_name:
                        return {"result": self.set_database(db_name), "action": "switch_database"}
                    else:
                        return {"error": "No database name provided"}
                
                elif intent.get("action") == "query":
                    collection_name = intent.get("collection_name")
                    if collection_name and collection_name in available_collections:
                        return {"result": self.execute_natural_language_query(collection_name, user_input), 
                                "action": "query", 
                                "collection": collection_name}
                    else:
                        # If no valid collection is identified, use the first available one
                        if available_collections:
                            self.console.print(f"[yellow]No valid collection identified. Using '{available_collections[0]}' instead.[/]")
                            return {"result": self.execute_natural_language_query(available_collections[0], user_input),
                                    "action": "query",
                                    "collection": available_collections[0]}
                        else:
                            return {"error": "No collections available in the current database"}
                
                else:
                    return {"error": "Unknown action requested"}
                
            except json.JSONDecodeError:
                return {"error": "Failed to parse LLM response as JSON"}
        else:
            return {"error": "No valid JSON found in LLM response"}

    def display_results(self, response):
        """Pretty print results using Rich"""
        if "error" in response:
            self.console.print(Panel(f"[bold red]{response['error']}[/]", title="Error", border_style="red"))
            return

        result = response.get("result", [])
        action = response.get("action", "")

        if action == "list_collections":
            if not result:
                self.console.print("[yellow]No collections found in this database[/]")
                return
                
            table = Table(title="Collections", show_header=True, header_style="bold magenta")
            table.add_column("Collection Name", style="cyan")
            
            for collection in result:
                table.add_row(collection)
            
            self.console.print(table)
            
        elif action == "switch_database":
            self.console.print(Markdown(f"### {result}"))
            
        elif action == "query":
            if not isinstance(result, list):
                self.console.print(result)
                return
                
            if not result:
                self.console.print(Panel("[yellow]No results found[/]", title="Query Results", border_style="yellow"))
                return
                
            self.console.print(f"\n[bold green]Found {len(result)} results:[/]")
            
            # Create a table for the results if they have a consistent structure
            first_doc = result[0]
            if isinstance(first_doc, dict):
                # Try to create a table with all fields
                table = Table(title=f"Results from {response.get('collection', 'Collection')}", 
                             show_header=True, header_style="bold magenta", 
                             show_lines=True)
                            
                # Add columns based on the first document's keys
                keys = list(first_doc.keys())
                for key in keys:
                    if key != "_id":  # Skip MongoDB internal ID for cleaner output
                        table.add_column(key, overflow="fold")
                
                # Add rows for each document
                for doc in result:
                    row_values = []
                    for key in keys:
                        if key != "_id":
                            value = doc.get(key, "")
                            # Make sure value is string and truncate if too long
                            str_value = str(value)
                            if len(str_value) > 50:
                                str_value = str_value[:47] + "..."
                            row_values.append(str_value)
                    
                    if row_values:
                        table.add_row(*row_values)
                
                self.console.print(table)
                
            # Also show individual documents in JSON format for complete view
            for i, item in enumerate(result):
                if "_id" in item:
                    # Convert ObjectId to string for JSON serialization
                    item["_id"] = str(item["_id"])
                    
                self.console.print(Panel(
                    Syntax(json.dumps(item, ensure_ascii=False, indent=2), "json", theme="monokai"),
                    title=f"[bold]Result {i+1}[/]",
                    border_style="green"
                ))

def main():
    """Main function to run the MongoDB agent"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]MongoDB Agent[/] with [bold green]Ollama Integration[/]",
        border_style="cyan"
    ))
    
    # Get MongoDB connection details
    mongo_uri = Prompt.ask("Enter MongoDB URI", default="mongodb://localhost:27017/")
    db_name = Prompt.ask("Enter database name", default="test")
    
    # Initialize agent
    try:
        agent = MongoDBAgent(mongo_uri, db_name)
        console.print(f"\n[bold green]Connected to database:[/] {db_name}")
        
        collections = agent.list_collections()
        if collections:
            table = Table(title="Available Collections", show_header=True, header_style="bold magenta")
            table.add_column("Collection Name", style="cyan")
            for collection in collections:
                table.add_row(collection)
            console.print(table)
        else:
            console.print("[yellow]No collections found in this database[/]")
        
        # Set Ollama model
        agent.model = Prompt.ask("\nEnter Ollama model to use", default="gemma3:12b")
        
        # Main interaction loop
        console.print(Panel(
            "[bold]Enter your questions about the database (or 'exit' to quit)[/]\n" + 
            "Example: 'Show me the first 5 items' or 'List all collections'",
            title="Instructions",
            border_style="green"
        ))
        
        while True:
            user_input = Prompt.ask("\n[bold cyan]Query[/]")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
                
            # Remove quotes if user included them
            user_input = user_input.strip("'\"")
            
            try:
                start_time = time.time()
                response = agent.process_user_input(user_input)
                elapsed_time = time.time() - start_time
                
                agent.display_results(response)
                console.print(f"[dim]Query processed in {elapsed_time:.2f} seconds[/]")
                    
            except Exception as e:
                console.print(Panel(f"[bold red]Error:[/] {str(e)}", title="Error", border_style="red"))
                console.print("[yellow]Try a different query or check your MongoDB connection[/]")
                
    except Exception as e:
        console.print(Panel(f"[bold red]Failed to initialize MongoDB Agent:[/] {str(e)}", 
                          title="Initialization Error", border_style="red"))

if __name__ == "__main__":
    main()
