"""import asyncio

# Define an asynchronous function (coroutine)
async def greet(name, delay):
    await asyncio.sleep(delay)  # Simulate I/O operation
    print(f"Hello, {name}! (after {delay} seconds)")

# Main coroutine
async def main():
    # Schedule multiple coroutines to run concurrently
    task1 = asyncio.create_task(greet("Alice", 2))
    task2 = asyncio.create_task(greet("Bob", 1))
    task3 = asyncio.create_task(greet("Charlie", 3))
    
    # Wait for all tasks to complete
    await task1
    await task2
    await task3

# Run the main coroutine
asyncio.run(main())"""


import asyncio
import aiohttp  # You'll need to install: pip install aiohttp

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = [
        'https://python.org',
        'https://httpbin.org/get'
    ]
    
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    for url, content in zip(urls, results):
        print(f"Fetched {url}, length: {len(content), content[:100]}...")  # Print first 100 characters
        print(f"Fetched {url}, length: {len(content)} characters")

asyncio.run(main())