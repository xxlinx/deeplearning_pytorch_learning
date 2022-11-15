class Person:
    def __call__(self, name):
        print("__call__" + "hello" + name)

    def hell(self,name):
        print("hello" + name)

person = Person()
person('zhangsan')
person.hell('lisi')