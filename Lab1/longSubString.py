def longestNonRepeatingSubstring(s):
    '''
    Finds the longest substring in a string with no repeating characters.
    If there is a tie, it will favor the one it found first.
    '''
    longestSubString = ""
    currentSubString = ""

    for char in s:
        
        if char not in currentSubString: #if the character isn't in this substring already, (no repeating characters) put it in the current substring
            currentSubString+=char
        
        else: #if it is in the current substring, we have to start over.
            if len(currentSubString)>len(longestSubString): #if the substring we just found is longer than the longest substring we've found so far, longest = current
                longestSubString = currentSubString
            currentSubString = char #start over from current character

    #in case the string didn't end in a repepeating character, do one more check to see if we found a longer string on the last substring.
    if len(currentSubString)>len(longestSubString):
                longestSubString = currentSubString
    return longestSubString

s = longestNonRepeatingSubstring("pwwkew")
print(s,len(s))
