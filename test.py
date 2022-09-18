import json

entity = []
label = []

with open('./data/test1.txt', 'r', encoding='utf-8') as f:
    for data in f.readlines():
        data = data.strip('\n')  # 去除文本中的换行符
        if data.startswith(u'\ufeff'):
            data = data.encode('utf8')[3:].decode('utf8')
        data1 = json.loads(data)
        print(data1["entities"])
        # print(data1)