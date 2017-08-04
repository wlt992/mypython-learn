#/usr/bin/env python
# 5-2
def prod(a,b):
    return a*b

# 5-3
def ceping():
    n = raw_input('pls input a number:')
    while not n.replace('.', '', 1).isdigit():
        print n, "is not a number"
        n = raw_input("pls input a number")
    # try:
    #    float(n)
    # except e:
    #     print e # can't input again

    n = float(n)
    while n > 100 or n < 0:
#        print n, "is not between 0 and 100"
#        n = raw_input("pls input a number")
        return
    if n>=90 and n<=100:
        return 'A'
    elif n >= 80:
        return 'B'
    elif n >= 70:
        return 'C'
    elif n >= 60:
        return 'D'
    else:
        return 'F'

def main():
    print "prod(3,4) is", prod(3,4)
    print ceping()

if __name__  == '__main__':
    main()
