
def check_sort(arr):
    for i,n in enumerate(arr):
        if (i != n-1):
            return False
    
    return True
    
def recursive(src, k, depth, count, min_count, max_depth):
    if depth == max_depth:
        return min_count
    
    if count > min_count:
        return min_count
    
    if check_sort(src):
        if min_count > count:
            min_count = count
        return min_count
        
    depth += 1
    for ci, n in enumerate(src):
        ti = n - 1
        if ci == ti:
            continue
        arr = src.copy()
        if abs(ci - ti) <= k:
            arr[ci], arr[ti] = arr[ti], arr[ci]
        else:
            if ci < ti:
                arr[ci], arr[ci+k] = arr[ci+k], arr[ci]
            elif ci > ti:
                arr[ci], arr[ci-k] = arr[ci-k], arr[ci]
    
        min_count = recursive(arr, k, depth, count+1, min_count, max_depth)
    return min_count
                
        
        
    
    


def solution(arr, k):
    answer = 0
    min_count = recursive(arr, k, 0, 0, 100, 1000)
    
    print(min_count)
    
    return min_count


if "__main__" == __name__:
     print(solution([4,5,2,3,1],2)==4)
     print(solution([5,4,3,2,1],4)==2)
     print(solution([5,4,3,2,1],2)==4)