#/usr/bin/env python
import random 
def main():
    choice = """
(J)iandao
(S)hitou
(B)u
(Q)uit

Enter your choice:"""
    
    while True:
        while True:
            try:
                s = raw_input(choice).strip()[0].lower()
            except (EOFError, KeyboardInterrupt, IndexError):
               # print 'Encounter an Interrupt.Ready to exit...'
                s = 'q'
            
            print "\nYou picked: [%s]" % s

            if s not in 'jsbq':
                print 'Invaid option, try again!'
            else:
                break
        if s == 'q':
            break
        
        choice_list = 'jsb'
        c = random.choice(choice_list)
        print "Computer's choice is: [%s]" % c

        l = 'Sorry, U Lose'
        w = 'Well done, U Win'
        d = 'Draw'
        judgement = {('j', 's'): l,
                ('s', 'j'): w,
                ('s', 'b'): l,
                ('b', 's'): w,
                ('b', 'j'): l,
                ('j', 'b'): w,
                ('j', 'j'): d,
                ('s', 's'): d,
                ('b', 'b'): d}
        result = judgement[s, c]
        print result
        
if __name__ == '__main__':
    main()

