#/usr/bin/env python

db = {}
def newuser():
    prompt = 'login desired: '
    while True:
        name = raw_input(prompt)
        if name in db:
            prompt = 'name taken, try another: '
            continue
        else:
            break

    pwd = raw_input('passwd: ')
    db[name] = pwd

def olduser():
    while True:
        name = raw_input('login: ')
        pwd = raw_input('passwd: ')
        password = db.get(name)
        if password == pwd:
            print 'welcome back', name
            break
        else:
            print 'login incorrect'
            continue

def showmenu():
    prompt = """
(N)ew User Login
(O)ld User Login
(Q)uit

Enter choice: """
    CMDs = {'n': newuser, 'o': olduser}
    while True:
        try:
            choice = raw_input(prompt).strip()[0].lower()
        except (EOFError, KeyboardInterrupt):
            choice = 'q'

        print '\nYou picked: [%s]' % choice

        if choice not in 'noq':
            print 'Invalid option, try again!'
            continue
        elif choice == 'q':
            break
        else:
            CMDs[choice]()

if __name__ == '__main__':
    showmenu()


