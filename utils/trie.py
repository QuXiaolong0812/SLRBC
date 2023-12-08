import collections
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True


    def enumerateMatch(self, word, space="_", backward=False):  #space=‘’
        matched = []
        #依次缩短word的长度，是词汇就保留，比如“中国语言”，依次判断“中国语言”，“中国语”，“中国”，“中”是否是词汇，是就加入到matched里面
        while len(word) > 0:
            if self.search(word):#判断word是否是词典里的一个词汇
                matched.append(space.join(word[:]))
            del word[-1]
        return matched

