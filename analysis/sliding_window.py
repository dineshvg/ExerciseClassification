## Source : https://coderwall.com/p/zvuvmg/sliding-window-in-python
def window(iterable, size=2):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    # yield win
    print win
    for e in i:
        win = win[1:] + [e]
        # yield win
        print win


## Source : https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/

def slidingWindow(sequence, winSize, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) / step) + 1
    print numOfChunks
    # Do the work
    for i in range(0, numOfChunks * step, step):
        height = len(ims)
        for j in range(0,height):
            list = ims[j]
            print list[i:i + winSize]

## Source : http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

sequence = [1, 2, 3, 4, 5, 6]
# slidingWindow(sequence,3,1)

ims = [[-99.99999657, -99.99999657, -99.99999654 , -96.55128326, -99.28115518, -99.67205004],
 [-99.99999021, -99.99999021, -99.9999902  , -96.94050894, -99.51976989, -99.85652882],
 [-99.99999695, -99.99999695, -99.99999694 , -98.52195776, -99.34765337, -99.73673601],
 [-99.99999408, -99.99999407, -99.99999407 , -98.3802451,  -99.70623551, -99.94709588],
 [-99.99999988, -99.99999988, -99.99999986 , -98.15567974, -99.45927961, -99.71954078],
 [-99.99999117, -99.99999034, -99.99998805 , -99.14954319, -99.74638024, -99.96497082]]

slidingWindow(ims[0],3,1)