def find_duplicates(arr):
    if not arr:
        return None
    
    for num in arr:
        if arr.count(num) > 1:
            return num
    
    return None

print(find_duplicates([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))