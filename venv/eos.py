import json
import sys

with open('/Users/priyanka/Downloads/eos/eos-usd.json') as file:
    ticks = json.load(file)

trade_id=[]
time=[]
side=[]
last_size=[]
best_bid=[]
best_ask=[]
price=[]
product_id=[]

for i in ticks:
    trade_id1, time1,side1,last_size1,best_bid1,best_ask1,price1,product_id1 = list(i.values())
    trade_id.append(int(trade_id1))
    time.append(time1)
    side.append(side1)
    last_size.append(float(last_size1))
    best_bid.append(float(best_bid1))
    best_ask.append(float(best_ask1))
    price.append(float(price1))
    product_id.append(product_id1)




date = sys.argv[1]

#sort the lists
zipped_lists = zip(trade_id, time,side,last_size,best_bid,best_ask,price,product_id)
sorted_lists = list(sorted(zipped_lists))


#subset of lists to get the records of the user entered date
filtered = (filter(lambda elems: date in elems[1], sorted_lists))
trade_id, time,side,last_size,best_bid,best_ask,price,product_id = [ list(tuple) for tuple in  zip(*filtered)]

values =[]
values.append(price[0])
values.append(max(price))
values.append(min(price))
values.append(price[-1])

print(values)

