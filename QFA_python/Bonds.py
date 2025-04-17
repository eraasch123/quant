from math import exp as e

class ZeroCouponBond:

    def __init__(self, principal, maturity, interest_rate):
        self.principal = principal
        self.maturity = maturity
        self.interest_rate = interest_rate / 100

    def present_value(self, x, n):
        return x / (1+self.interest_rate)**n
    
    def calc_price(self):
        return self.present_value(self.principal, self.maturity)
    
class CouponBond:

        def __init__(self, principal, rate, maturity, interest_rate):
             self.principal = principal
             self.rate = rate / 100
             self.maturity = maturity
             self.interest_rate = interest_rate / 100
        
        def present_value(self, x, n):
             return x / (1 + self.interest_rate)**n

        def calc_price(self):
             price = 0
             for t in range(1, self.maturity+1):
                  price = price + self.present_value(self.principal * self.rate, t)
             
             price = price + self.present_value(self.principal, self.maturity)

             return price

class ZeroCouponBondContinuous:

    def __init__(self, principal, maturity, interest_rate):
        self.principal = principal
        self.maturity = maturity
        self.interest_rate = interest_rate / 100

    def present_value(self, x, n):
        return x * e(-self.interest_rate*n)
    
    def calc_price(self):
        return self.present_value(self.principal, self.maturity)
    
class CouponBondContinuous:

        def __init__(self, principal, rate, maturity, interest_rate):
             self.principal = principal
             self.rate = rate / 100
             self.maturity = maturity
             self.interest_rate = interest_rate / 100
        
        def present_value(self, x, n):
             return x * e(-self.interest_rate*n)

        def calc_price(self):
             price = 0
             for t in range(1, self.maturity+1):
                  price = price + self.present_value(self.principal * self.rate, t)
             
             price = price + self.present_value(self.principal, self.maturity)

             return price

if __name__ == '__main__':
    # Discrete Bonds
    bond = ZeroCouponBond(1000, 2, 4)
    print("The price of the zero coupon bond is $%.2f" % bond.calc_price())
    Cbond = CouponBond(1000, 10, 3, 4)
    print("The price of the coupon bond is $%.2f" % Cbond.calc_price())

    # Continuous Bonds
    bond_cont = ZeroCouponBondContinuous(1000, 2, 4)
    print("The price of the continuous zero coupon bond is $%.2f" % bond_cont.calc_price())
    Cbond_cont = CouponBondContinuous(1000, 10, 3, 4)
    print("The price of the continuous coupon bond is $%.2f" % Cbond_cont.calc_price())
