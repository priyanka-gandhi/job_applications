import sys
with open('/Users/priyanka/Downloads/eos/eos-usd.json', 'r') as jsonFile:
    json_obj = jsonFile.readlines()


trade_id=[]
time=[]
side=[]
last_size=[]
best_bid=[]
best_ask=[]
price=[]
product_id=[]

#Creating lists to avoid using Pandas for reading and storing data
for i in json_obj[1:-1]:
    i = (i.rstrip()).lstrip()[:-1]
    i=i.strip('"')
    i=i.replace('": "',"")
    if "trade_id" in i:
        i=i.strip('trade_id: ')
        trade_id.append(int(i))
    elif "time" in i:
        i=i.strip('time: ')
        time.append(i)
    elif "side" in i:
        i=i.replace('side',"")
        side.append(i)
    elif "last_size" in i:
        i=i.strip('last_size: ')
        last_size.append(float(i))
    elif "best_bid" in i:
        i=i.strip('best_bid: ')
        best_bid.append(float(i))
    elif "best_ask" in i:
        i=i.strip('best_ask: ')
        best_ask.append(float(i))
    elif "price" in i:
        i=i.strip('price: ')
        price.append(float(i))
    elif "product_id" in i:
        i=i.strip('product_id: ')
        product_id.append(i)
    else:
        pass


date = sys.argv[1]

#sort the lists
zipped_lists = zip(trade_id, time,side,last_size,best_bid,best_ask,price,product_id)
sorted_lists = sorted(zipped_lists)

#subset of lists to get the records of the user entered date
filtered = (filter(lambda elems: date in elems[1], sorted_lists))
trade_id, time,side,last_size,best_bid,best_ask,price,product_id = [ list(tuple) for tuple in  zip(*filtered)]

values =[]
values.append(price[0])
values.append(max(price))
values.append(min(price))
values.append(price[-1])

print(values)



