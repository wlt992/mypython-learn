#/usr/bin/env python
import time,base64

db = {}
def newuser():
    prompt = 'login desired: '
    name = raw_input(prompt).lower().strip()
    #while True:
    #    name = raw_input(prompt).lower()
        #if name in [x.lower() for x in list(db.keys())]:
        #    prompt = 'name taken, try another: '
        #    continue
        #else:
        #    break

    pwd = raw_input('passwd: ')
    pwd = base64.encodestring(pwd)
    db[name] = [pwd, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())]
    print 'Create user [%s]' % name

def olduser():
    name = raw_input('login: ').lower().strip()
    pwd = raw_input('passwd: ')
    pwd = base64.encodestring(pwd)
    if name in db:
        try:
            password = db[name][0]
        except TypeError:
            print 'login incorrect'

        if password == pwd:
            print 'welcome back', name
            
            timeNow = time.strftime('%Y-%m-%d %H:%M:%S',
                time.localtime())
            try:
                timeBefore = db[name][1]
                timeNow_stamp = time.mktime(time.strptime(timeNow, '%Y-%m-%d %H:%M:%S'))
                timeBefore_stamp = time.mktime(time.strptime(timeBefore, '%Y-%m-%d %H:%M:%S')) 
                if timeNow_stamp - timeBefore_stamp <= 4*60*40:
                    print 'You already logged in at : %s' % timeBefore
                db[name][1] = timeNow

            except TypeError:
                del db[name]
                db[name] = [password, timeNow]
        else:
            print 'login incorrect'
    else:
        print 'User [%s] doesnot exists, Ready to create...' % name
        newuser()

def showusers():
    print 'all users below:\n'
    for i in db.keys():
        print 'name: %s, pwd: %s, lastlogin: %s' % (i, db[i][0], db[i][1])

def deluser():
    name = raw_input('Enter name you want del:')
    if name in db:
        print 'You want del [%s]' % name
        del db[name]
    else:
        print '[%s] not exists in db' % name
    

def showmenu():

    prompt = """
(U)ser Login
(S)how users
(D)el user
(Q)uit

Enter choice: """
    CMDs = {'u': olduser, 's': showusers, 'd': deluser}
    while True:
        try:
            choice = raw_input(prompt).strip()[0].lower()
        except (EOFError, KeyboardInterrupt):
            choice = 'q'

        print '\nYou picked: [%s]' % choice

        if choice not in 'usdq':
            print 'Invalid option, try again!'
            continue
        elif choice == 'q':
            break
        else:
            CMDs[choice]()
            print '-'*30

if __name__ == '__main__':
    showmenu()


