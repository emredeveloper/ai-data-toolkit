import json
from datasets import load_dataset
import warnings

def load_seed_tools():
    """Veri setini Hugging Face'den yükler, önbelleği devre dışı bırakır."""
    try:
        # Önbelleği devre dışı bırakmak için cache_dir=None kullanıyoruz
        dataset = load_dataset("argilla-warehouse/python-seed-tools", split="train", cache_dir=None)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Please check your internet connection or try again later.")
        return None

def find_function(dataset, func_name):
    """Veri setinde verilen fonksiyon adını arar."""
    if dataset is None:
        return None
    for item in dataset:
        if item["func_name"].lower() == func_name.lower():
            return item
    return None

def parse_tools(tools_str):
    """Tools JSON string'ini parse eder ve parametreleri döndürür."""
    try:
        tools = json.loads(tools_str)
        if not tools or "function" not in tools[0]:
            return None
        
        function_info = tools[0]["function"]
        params = function_info.get("parameters", {}).get("properties", {})
        required = function_info.get("parameters", {}).get("required", [])
        
        param_details = []
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "unknown")
            param_desc = param_info.get("description", "No description")
            is_required = param_name in required
            param_details.append({
                "name": param_name,
                "type": param_type,
                "description": param_desc,
                "required": is_required
            })
        
        return param_details
    except json.JSONDecodeError:
        return None

def display_function_info(func_info):
    """Fonksiyon bilgilerini kullanıcı dostu bir şekilde görüntüler."""
    print(f"\nFunction: {func_info['func_name']}")
    print(f"Description: {func_info['func_desc']}\n")
    
    params = parse_tools(func_info["tools"])
    if not params:
        print("No parameters found.")
        return
    
    print("Parameters:")
    for param in params:
        required_str = " (Required)" if param["required"] else ""
        print(f"- {param['name']} ({param['type']}){required_str}: {param['description']}")

def main():
    """Ana uygulama döngüsü."""
    print("Welcome to Python Function Lookup!")
    print("Enter a function name to see its details (or 'quit' to exit).")
    
    # Hugging Face uyarılarını bastır
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    
    dataset = load_seed_tools()
    if dataset is None:
        print("Exiting due to dataset loading failure.")
        return
    
    while True:
        func_name = input("\nFunction name: ").strip()
        if func_name.lower() == "quit":
            print("Goodbye!")
            break
        
        func_info = find_function(dataset, func_name)
        if func_info:
            display_function_info(func_info)
        else:
            print(f"Function '{func_name}' not found. Try another name.")

if __name__ == "__main__":
    main()