import matplotlib.pyplot as pyplot
from IPython import display

# Enable interactive plotting.
pyplot.ion()

def plot(scores, mean_scores):
    # Update and display the plot with new score data.
    display.clear_output(wait=True)
    display.display(pyplot.gcf())
    pyplot.clf()
    pyplot.title('Training...')
    pyplot.xlabel('Number of Games')
    pyplot.ylabel('Score')
    pyplot.plot(scores)
    pyplot.plot(mean_scores)
    pyplot.ylim(ymin=0)

    # Annotate the final score and mean score values.
    pyplot.text(len(scores) - 1, scores[-1], str(scores[-1]))
    pyplot.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))