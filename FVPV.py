from math import exp as e

def future_discrete_value(x, r, n):
    return(x*(1+r)**n)

def present_discrete_value(x, r, n):
    return(x*(1+r)**-n)

def future_continuous_value(x, r, t):
    return(x*e(r*t))

def present_continuous_value(x, r, t):
    return(x*e(-r*t))

if __name__ == '__main__':

    # Value of investment in dollars
    x = 100

    # Interest rate
    r = .05

    # Time in years
    n = 5

    print("Future discrete value %s" % future_discrete_value(x, r, n))
    print("Present discrete value %s" % present_discrete_value(x, r, n))
    print("Future continuous value %s" % future_continuous_value(x, r, n))
    print("Present continious value %s" % present_continuous_value(x, r, n))