import matplotlib.pyplot as plt
import numpy as np

VARS = 'xyzwuvpqij'

def poly_to_str(polynomial):
    '''
    Convert the polynomial to a string
    
    polynomial:   a list of (coef, [monomial])
    '''
    
    return ' + '.join(['%s%s' % (poly[0], ''.join(['%s^%s' % (VARS[i], poly[1][i]) for i in range(len(poly[1]))])) for poly in polynomial])

def poly_to_func(polynomial):
    '''
    Convert a polynomial to a func
    
    polynomial:   a list of (coef, [monomial])
    '''
    
    def func(x):
        return sum(mono[0] * np.product(np.power(x, mono[1])) for mono in polynomial)
    
    return func

def plot_1d(polynomial, bounds, point=None, constraints=None):
    '''
    Plot a function f(x) which is a univariate polynomial and
    add an optional point to the plot
    
    polynomial:   a polynomial defined as a list of (coef, [monomial])
    bounds:       the domain
    point:        an optional point
    '''

    # generate a range of x values and compute y values
    fx = np.vectorize(poly_to_func(polynomial))
    x = np.linspace(*bounds, 400)
    y = fx(x)
    
    # plot constraint area
    if constraints is not None and len(constraints) > 0:
        
        # this is the bar we will plot
        miny = min(y) - 0.1
        
        # plot each constraint
        for const in constraints:
            cx = np.vectorize(poly_to_func(const))
            const_y = cx(x)
            bad_area_x = x[const_y < 0]
            plt.plot(bad_area_x, [miny] * len(bad_area_x), 'r')

    # plot the function
    plt.plot(x, y)

    # plot a single point in red
    if point is not None:
        plt.scatter(point, fx(point), color='red')

    # display the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(poly_to_str(polynomial))
    plt.grid(True)
    plt.show()
