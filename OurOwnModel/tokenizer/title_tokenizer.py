import re


# 定义tokenize方法：
def title_tokenize(text):
    # 定义分割符，该变量作为超参数，可以根据情况删减
    filters = ['#', '\$', '@', '&', '!',
                '\t', '\n', '\x97', '\x96', 
                ',', '\(', '\)','\[','\]','/',
                ';']

    # 运用正则进行分词:
    # 去除一些特殊记号：
    text = re.sub("&quot", " ", text, flags=re.S)
    text = re.sub("=", " ", text, flags=re.S)
    text = re.sub(":", " ", text, flags=re.S)
    text = re.sub("\s:\s", " ", text, flags=re.S)
    text = re.sub("&amp", " ", text, flags=re.S)
    text = re.sub("AT&amp;T", "brand ATT", text, flags=re.S)
    text = re.sub("T-Mobile", "brand tmobile", text, flags=re.S)
    text = re.sub("\.+", " ", text, flags=re.S)
    # 将网页内容如<>以及分隔符全部替代为空格
    text = re.sub("<.*?>", " ", text, flags=re.S)
    # 将n't转换为not
    text = re.sub("n't", " not", text, flags=re.S)
    # 删去单独出现的-
    text = re.sub("\s-+\s", " ", text, flags=re.S)
    text = re.sub("\s-+$", " ", text, flags=re.S)
    text = re.sub("^-+\s", " ", text, flags=re.S)
    # 将百分数转化：
    text = re.sub("([0-9]+)%", "\g<1> % ", text, flags=re.S)
    # 删除句号并不影响小数
    text = re.sub("\. ", "  ", text, flags=re.S)
    text = re.sub("\.$", "  ", text, flags=re.S)
    text = re.sub("|".join(filters), "  ", text, flags=re.S)

    # 将*删去：
    text = re.sub(" \*([0-9a-zA-Z]+) ", " \g<1> ", text, flags=re.S)
    text = re.sub("\*([0-9a-zA-Z]+) ", "\g<1> ", text, flags=re.S)
    text = re.sub(" \*([0-9a-zA-Z]+)", " \g<1>", text, flags=re.S)

    text = re.sub(" ([0-9a-zA-Z]+)\* ", " \g<1> ", text, flags=re.S)
    text = re.sub("([0-9a-zA-Z]+)\* ", "\g<1> ", text, flags=re.S)
    text = re.sub(" ([0-9a-zA-Z]+)\*", " \g<1>", text, flags=re.S)

    text = re.sub("\*([0-9a-zA-Z]+)\* ", "\g<1> ", text, flags=re.S)
    text = re.sub(" \*([0-9a-zA-Z]+)\* ", " \g<1> ", text, flags=re.S)  
    text = re.sub(" \*([0-9a-zA-Z]+)\*", " \g<1>", text, flags=re.S)
    
    # 将加号单独列出：
    text = re.sub("(.)\+(.)", "\g<1> + \g<2>", text, flags=re.S)
    # 将不是型号且有-和单词的列出：
    text = re.sub("\s([a-zA-Z]+)-([a-zA-Z]+)\s", " \g<1> \g<2> ", text, flags=re.S)
    text = re.sub("\s([a-zA-Z]+)-([a-zA-Z]+)-([a-zA-Z]+)\s", " \g<1> \g<2> \g<3> ", text, flags=re.S)

    text = re.sub("\s-+([0-9a-zA-Z]+)\s", " \g<1> ", text, flags=re.S)
    text = re.sub("\s([0-9a-zA-Z]+)-+\s", " \g<1> ", text, flags=re.S)
    text = re.sub("\s([0-9a-zA-Z]+)-+$", " \g<1>", text, flags=re.S)
    text = re.sub("\s-+([0-9a-zA-Z]+)", " \g<1>", text, flags=re.S)
    # text = re.sub("-([a-zA-Z]+)\s", "\g<1> ", text, flags=re.S)
    # text = re.sub("\s-([a-zA-Z]+)", " \g<1>", text, flags=re.S)
    # 将分隔符全部替代为空格
    # text = re.sub("|".join(filters), " ", text, flags=re.S)
    # 运用strip()返回移除字符串头尾指定的字符（默认为空格或换行符）生成的新字符串:并转换为小写：
    result = [i.strip().lower() for i in text.split()]
    return result
