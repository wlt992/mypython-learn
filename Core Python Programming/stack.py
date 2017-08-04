#/usr/bin/env python

stack = []
def pushit():
    stack.append(raw_input(' Enter New String: ').strip())

def popit():
    if len(stack) == 0:
        print 'Cannot pop from an empty stack!'
    else:
        t = stack.pop()
        print 'Remove [' + t + ']'

def viewstack():
    print stack

CMDs = {'u': pushit, 'o': popit, 'v': viewstack} # link to function

def showmenu():
    pr = """
 p(U)sh
 p(O)p
(V)iew
(Q)uit

Enter choice: """
    
    while True:
        while True:
            try:
                choice = raw_input(pr).strip()[0].lower()
            except (EOFError, KeyboardInterrupt, IndexError):
                print '\nEncounter Interrupt, Ready to exit...'
                choice = 'q'

            if choice not in 'uovq':
                print 'invalid option, try again!'
            else:
                break
        if choice == 'q':
            break
        CMDs[choice]()

if __name__ == '__main__':
    showmenu()


