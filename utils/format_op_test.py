from format_op import params2id, listformat

def test1():
    '''
    Results:
        machine_learning
    '''
    print(params2id('machine', 'learning'))

def test2():
    '''
    Results :
        a : [0.1, 0.2, 1.0]
        listformat(a) : 0.1/0.2/1.0
    '''
    a = [0.1, 0.2, 1.00]
    print(" a : {}".format(a))
    print(" listformat(a) : {}".format(listformat(a)))

if __name__=='__main__':
    test1()
    test2()
