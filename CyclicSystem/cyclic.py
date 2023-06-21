import sys
sys.path.insert(0, '../');

from root import *

# Main execution block.
if __name__ == '__main__':
    x0 = np.array( [[7.5],[0],[0]] );
    simulateModelWithControl( x0, model, control, N=1000 );
    print("Animation finished...")