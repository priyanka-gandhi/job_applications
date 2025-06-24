

#Permutations Solution 1 -- naive
s="ab"

for i in range(len(s)):
    for j in range(len(s)):
        if i!=j:
            print(s[i]+s[j])
print()
#Permutations Solution 2 -- Function to generate permutations

def permute(inp):
    n = len(inp)
    mx = 2**n       # Number of permutations is 2^n  pow(2,n)
    print(mx)
    inp = inp.lower()   # Converting string to lower case
    for i in range(mx):
        combination = [k for k in inp]      # If j-th bit is set, we convert it to upper case
        for j in range(n):
            #print(i,"   ",j,"       ",(i // (2**j)),"      ",(i // (2**j))&1)
            if ((i // (2**j)) & 1) == 1:            #i>>j
                combination[j] = inp[j].upper()
        temp = ""
        for j in combination:
            temp += j
        print(temp)
permute("ABC")

print()

#You are provided a set of positive integers (an array of integers). Each integer represents a number of nights users request on Airbnb.com. If you are a host, you need to design and implement an algorithm to find out the maximum number of nights you can accommodate. The constraint is that you have to reserve at least one day between each request, so that you have time to clean the room.
def rob(nums):
    if nums == None : return 0
    n = len(nums)
    if n == 0: return 0
    if (n == 1):  return nums[0]
    f1 = nums[0] # max of far, excluding current
    f2 = max(nums[0], nums[1]) # max so far
    for i in range(2,n):
        f = max(f1 + nums[i], f2)
        #print(f,"   ",f1,"  ",f2)
        f1 = f2
        f2 = f
    return (f2)

print(rob([5, 1, 2, 6, 20, 2]))

