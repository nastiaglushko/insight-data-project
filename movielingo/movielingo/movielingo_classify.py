import plotly.graph_objects as go
import plotly.express as px

def get_result(movie_instance, l2_level):
    ''' Based on language complexity composite score + heuristics, produces
    actionable feedback for language learner re: whether the movie is good for them
    :param: list of proficiency labels with corresponding ratio of text:
    e.g., [["A1", 56], ["B2", 44]], language proficiency level ('BegInter' / 'UpperInterAdv')
    :results: " (movie X) is just right for you!" etc.
    '''
    labels = []
    vals = []
    for i in range(len(results)):
        labels.append(results[i][0])
        vals.append(results[i][1])
    b1_can_understand = 0
    if 'A2' in labels:
        i = labels.index('A2')
        b1_can_understand += vals[i]
    if 'B1' in labels:
        i = labels.index('B1')
        b1_can_understand += vals[i]
    if l2_level == 'BegInter':
        if b1_can_understand >= 75:
            result = 'is just right for you!'
        elif b1_can_understand >= 50:
            result = 'might be a bit too difficult for you!'
        else:
            'is probably too difficult for you.'
    if l2_level == 'UpperInterAdv':
        if b1_can_understand >= 75:
            result = 'is almost too easy for you!'
        else:
            result = 'is just right for you!'
    return result

def plot_subtitle_difficulty(movie_instance):
    ''' Plot distribution of subtitle difficulty as pie chart
    :param: list of proficiency labels with corresponding ratio of text:
    e.g., [["A1", 56], ["B2", 44]], movie_title (str)
    :returns: plotly pie chart in html format
    '''
    labels = []
    vals = []
    difficulty_results = movie_instance.difficulty
    for i in range(len(difficulty_results)):
        labels.append(difficulty_results[i][0])
        vals.append(difficulty_results[i][1])
    fig = px.pie(labels=labels, values=vals, names = labels, hole=.3, color_discrete_sequence=px.colors.qualitative.G10)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    }, autosize = False, width = 400, height = 400)
    fig.update_layout({'margin': dict(l=0, r=0, t=50, b=50)},
                      {'font': {'family': 'Old Standard TT', 'size': 14, 'color': 'white'}},
                      showlegend=False)
    difficulty_plot = fig.to_html(full_html = True)
    return difficulty_plot