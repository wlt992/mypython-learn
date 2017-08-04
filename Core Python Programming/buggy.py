#/usr/bin/env python

flag = 0
while True:
    num_str = raw_input(" Enter a number: ")
    try:
        num_num = int(num_str)
        flag = 1
    except:
        print 'num_str is not a int number!'
    if flag == 1:
        break

fac_list = range(1, num_num + 1)

print "BEFORE:", fac_list

i = 0
a = [] 
while i < len(fac_list):
    if num_num % fac_list[i] == 0:
        a.append(fac_list[i])

    i = i + 1

for i in a:
    fac_list.remove(i)

print "AFTER:", fac_list

