#peak
#import required packages
#basics
import pandas as pd 
import numpy as np

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image


#nlp
import re    #for regex
import nltk
from nltk.corpus import stopwords
import gensim
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
eng_stopwords = set(stopwords.words("english"))


#stats
from statsmodels.stats.proportion import proportion_confint

#misc
import gc
import time
import warnings

#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
warnings.filterwarnings("ignore")
#import all the files!
train=pd.read_csv("../input/donorschoose-application-screening/train.csv")
resources=pd.read_csv("../input/donorschoose-application-screening/resources.csv")
test=pd.read_csv("../input/donorschoose-application-screening/test.csv")
sample_sub=pd.read_csv("../input/donorschoose-application-screening/sample_submission.csv")
# peak at the data
train.head()
#take a peak
resources.head()
#check test train split
nrow_train=train.shape[0]
nrow_test=test.shape[0]
sum=nrow_train+nrow_test
print("Checking proportion of Test-train split")
print("       : train  : test")
print("rows   :",nrow_train,":",nrow_test)
print("perc   :",round(nrow_train*100/sum),"    :",round(nrow_test*100/sum))
# check for missing values
print("Check for Percent of missing values in Train dataset")
null_check=train.isnull().sum()
(null_check/len(train))*100
# check for missing values
print("Check for Percent of missing values in RESOURCES file")
null_check=resources.isnull().sum()
(null_check/len(resources))*100
x=train.project_is_approved.value_counts()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Target Variable",fontsize=20)
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Project is approved?', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
print("Approval rate:",x[1]/(x[0]+x[1])*100)
# take IDs of the projects which have null description
print("There are",resources.description.isnull().sum(),"NULL entries in description column of Resources dataset")
null_ids=resources[resources.description.isnull()].id
print("Those Null entries are from",len(null_ids.unique()),"projects")
null_entries_train=train[train.id.isin(null_ids)]
print("There are",len(null_entries_train),"of these projects are in train")
null_entries_test=test[test.id.isin(null_ids)]
print("There are",len(null_entries_test),"of these projects are in test")
x=null_entries_train.project_is_approved.value_counts()
print("Approval rate of projects with NULL as the description under Resources:",x[1]/(x[0]+x[1])*100)
x=null_entries_train[null_entries_train.teacher_number_of_previously_posted_projects==0].project_is_approved.value_counts()
print("Approval rate of projects with NULL as the description under Resources and 0 previous project submissions:",x[1]/(x[0]+x[1])*100)
# null_entries_train[null_entries_train.project_is_approved==0].head(2)
# No obvious predictable pattern :(
null_entries_train[null_entries_train.project_is_approved==0].project_resource_summary.iloc[0]
null_entries_train[null_entries_train.project_is_approved==0].project_resource_summary.iloc[1]
#making this for easy subsetting later
approvals=train[train.project_is_approved==1]
rejects=train[train.project_is_approved==0]

# lets make a simple re-usable function to make the plots!
# This lets us add more functionality if needed later and it would be replicated across all plots!
def make_custom_plot(target_column_name='',title='',total_counts=None,approvals_counts=None,rejects_counts=None,x_rotation_angle=0):
    """        
    Description:
        Creates a 1x2 plot of 1. the # of projects across the variable and the 2. A stacked Percentage bar chart of the Approval rates across the variable
    Useage: 1) make_custom_plot('gender','Analyzing Gender')
            2) make_custom_plot(total_counts=x,approvals_counts=x1,rejects_counts=x2)
    """
    if(target_column_name!=''):
        x=train[target_column_name].value_counts()
        x1=approvals[target_column_name].value_counts()
        x2=rejects[target_column_name].value_counts()
    else:
        x=total_counts
        x1=approvals_counts
        x2=rejects_counts
        target_column_name=title
    #plot initiate
    plt.figure(figsize=(16,6))
    
    #super title
    plt.suptitle(title,fontsize=18)
    plt.subplot(121)
    #title and labels for plot1
    plt.title('Total Projects Submitted',fontsize=12)
    plt.ylabel('# of Projects', fontsize=12)
    plt.xlabel(target_column_name, fontsize=12)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=x_rotation_angle)
    # Barplot
    ax= sns.barplot(x.index, x.values, alpha=0.8)

    #adding the text labels
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    
    plt.subplot(122)
    #title and labels for plot2
    plt.title('Approval Rate',fontsize=12)
    plt.ylabel('Percent Approved Projects', fontsize=12)
    plt.xlabel(target_column_name, fontsize=12)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=x_rotation_angle)
    # https://python-graph-gallery.com/13-percent-stacked-barplot/
    r=np.arange(len(x))
    totals=x
    greenBars = [i / j * 100 for i,j in zip(x1, totals)]
    redBars = [i / j * 100 for i,j in zip(x2, totals)]

    barWidth = 0.85
    names = x.index
    # Create green Bars
    plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Approved")
    # Create red Bars
    plt.bar(r, redBars, bottom=greenBars, color='red', edgecolor='white', width=barWidth, label="Rejected")
    # Custom x axis
    plt.xticks(r, names)
    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    plt.show()
# Adding another custom function with Error bars

def make_custom_plot2(target_column_name='',title='',total_counts=None,approvals_counts=None,rejects_counts=None,x_rotation_angle=0):
    """        
    Description:
        Creates a 1x2 plot of 1. the # of projects across the variable(stacked bar chart) and the 2. Error bar using Binomial confidence interval
    Useage: 1) make_custom_plot('gender','Analyzing Gender')
            2) make_custom_plot(total_counts=x,approvals_counts=x1,rejects_counts=x2)
    """
    if(target_column_name!=''):
        x=train[target_column_name].value_counts()
        x1=approvals[target_column_name].value_counts()
        x2=rejects[target_column_name].value_counts()
    else:
        x=total_counts
        x1=approvals_counts
        x2=rejects_counts
        target_column_name=title
    #plot initiate
    plt.figure(figsize=(25,6))
    
    #super title
    plt.suptitle(title,fontsize=18)
    plt.subplot(121)
    #title and labels for plot1
    plt.title('Total Projects Submitted',fontsize=12)
    plt.ylabel('# of Projects', fontsize=12)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=x_rotation_angle)
    # https://python-graph-gallery.com/13-percent-stacked-barplot/
    r=np.arange(len(x))
    totals=x
    greenBars = x1
    redBars = x2

    barWidth = 0.85
    names = x.index
    # Create green Bars
    plt.bar(r, greenBars, color='#9AFF87', width=barWidth, label="Approved")
    # Create red Bars
    plt.bar(r, redBars, bottom=greenBars, color='#60B350',width=barWidth, label="Rejected")
    # Custom x axis
    plt.xticks(r, names)
    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    plt.subplot(122)
    #title and labels for plot2
    plt.title('Approval Rate(with Binomial Confidence intervals)',fontsize=12)
    plt.ylabel('Percent Approved Projects', fontsize=12)
    plt.xlabel(target_column_name, fontsize=12)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=x_rotation_angle)
    # https://python-graph-gallery.com/13-percent-stacked-barplot/
    top_list=[]
    bot_list=[]
    for n_total,n_approved in zip(x,x1):
        top_val,bot_val = proportion_confint(count=n_approved,nobs=n_total,alpha=0.05,method='normal')
        top_list.append(top_val)
        bot_list.append(bot_val)
    plt.errorbar(x=x.index, 
             y=x1/x, 
             yerr=[(top-bot)/2 for top,bot in zip(top_list,bot_list)],
             fmt='o',mfc='red',
             mec='green',mew=2,ms=5,capsize =10)
    plt.hlines(xmin=0, xmax=len(x)-1,
           y=x1.sum()/x.sum(), 
           linewidth=1.0,
           color="green")
    plt.show()

# Making a function instead of writing code for single text columns so that its more scalable and can be applied to all text cols
def get_text_stats(text_col):
    """
    Get Wordcount,Unique Wordcount and WordCount Percent and make appropriate visuals
    Todo: Add more text stats
    """
    title="Text Stats of " + text_col
    target_col='project_is_approved'
    # Borrowed from previous work at https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda
    text_stats=pd.DataFrame()
    text_stats[target_col]=train[target_col]
    #Word count 
    text_stats['word_count']=train[text_col].apply(lambda x: len(str(x).split()))
    #Unique word 
    text_stats['count_unique_word']=train[text_col].apply(lambda x: len(set(str(x).split())))
    #Word count percent in each comment:
    text_stats['word_unique_percent']=(text_stats['count_unique_word']*100)/text_stats['word_count']
    
    temp_df = pd.melt(text_stats, value_vars=['word_count', 'count_unique_word'], id_vars=target_col)
    
    print("------ Sample from an Approved project ------\n")
    print(approvals[text_col].iloc[0])
    
    print("\n------ Sample from a Rejected project ------\n")
    print(rejects[text_col].iloc[0])
    # Need to make this pythonic
    #get_binCI <- function(x,n) as.list(setNames(binom.test(x,n)$conf.int, c("lwr", "upr")))

    
    #plotting
    plt.figure(figsize=(16,5))
    plt.subplot(121)
    plt.suptitle(title,fontsize=16)
    #re-shaping as required
    plt.title("Word Count")
    sns.violinplot(x='variable', y='value', hue=str(target_col), data=temp_df,inner='quartile')
    plt.ylabel('# of projects', fontsize=12)
    
    plt.subplot(122)
    plt.title("Percentage of Unique words - effect on Approval")
    ax=sns.kdeplot(text_stats[text_stats.project_is_approved == 0].word_unique_percent, label="Not Approved",shade=True,color='r')
    ax=sns.kdeplot(text_stats[text_stats.project_is_approved == 1].word_unique_percent, label="Approved")
    plt.legend()
    plt.xlabel('Percent unique words', fontsize=12)
    plt.ylabel('# of projects', fontsize=12)
    plt.show()
# for the wordcloud
stopword=set(STOPWORDS)

# Custom Adding some stop words to make better and more meaningful wordclouds
stopword.add('will')
stopword.add('student')
stopword.add('students')
stopword.add('class')
stopword.add('classroom')
stopword.add('child')
stopword.add('children')
stopword.add('teacher')
stopword.add('school')
stopword.add('needs')

def make_word_clouds(text_col,approved_mask,rejected_mask,title_overall):
    """
    Makes two wordclouds : one for Approvals and one for rejects
    
    Todo: Think of faceted clouds across categorical var (Eg:Grade category)
    """
    plt.figure(figsize=(16,8))
    plt.suptitle(title_overall,fontsize=20)
    plt.subplot(121)
    # Get text col from approvals subset
    text=approvals[text_col].values
    # make wordcloud
    wc= WordCloud(background_color="white",max_words=1000,stopwords=stopword,mask=approved_mask,normalize_plurals=True)
    wc.generate(" ".join(text))

    plt.axis("off")
    plt.title("Words frequented in Approved Projects", fontsize=16)
    #https://matplotlib.org/examples/color/colormaps_reference.html for colormaps
    plt.imshow(wc.recolor(colormap='inferno',random_state=17), alpha=0.98,interpolation='bilinear')
    
    plt.subplot(122)
    # Get text col from Rejects subset
    text=rejects[text_col].values

    # make wordcloud
    wc= WordCloud(background_color="black",max_words=1000,stopwords=stopword,mask=rejected_mask,normalize_plurals=True)
    wc.generate(" ".join(text))
    
    plt.axis("off")
    plt.title("Words frequented in Rejected Projects", fontsize=16)
    plt.imshow(wc.recolor(colormap='Pastel1',random_state=17), alpha=0.98,interpolation='bilinear')
    plt.show()
make_custom_plot2('teacher_prefix','Does a Title affect approval?')
# Creating the gender column
gender_mapping = {"Ms.": "Female", "Mrs.":"Female", "Mr.":"Male", "Teacher":"Unknown", "Dr.":"Unknown", np.nan:"Unknown"  }
train["gender"] = train.teacher_prefix.map(gender_mapping)
approvals["gender"] = approvals.teacher_prefix.map(gender_mapping)
rejects["gender"] = rejects.teacher_prefix.map(gender_mapping)
test['gender'] = test.teacher_prefix.map(gender_mapping)
make_custom_plot('gender','Analyzing Gender')
make_custom_plot2('project_grade_category','Do smaller kids get more approval rates?')
x= train.project_subject_categories.value_counts()
#prep for chart
x=x.sort_values(ascending=False)
x=x.iloc[0:20]

#chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[0])
plt.title("What are the frequent subject categories?",fontsize=16)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# Projects', fontsize=12)
plt.xlabel('Subject Category', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
print("There are ",len(train.project_subject_categories.unique()),"unique Subject Categories")
# Grouping similar categories for overall
subject_cats=','.join(train['project_subject_categories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_categories']) # to split on ","
cats.project_subject_categories=cats.project_subject_categories.str.strip() # to remove unwanted spaces
x=cats.project_subject_categories.value_counts()
print("There are",len(x),"different subject categories after cleaning")

# repeat for approved group and rejected group
# Grouping similar categories for approved 
subject_cats=','.join(approvals['project_subject_categories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_categories']) # to split on ","
cats.project_subject_categories=cats.project_subject_categories.str.strip() # to remove unwanted spaces
x1=cats.project_subject_categories.value_counts()

# Grouping similar categories for rejected
subject_cats=','.join(rejects['project_subject_categories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_categories']) # to split on ","
cats.project_subject_categories=cats.project_subject_categories.str.strip() # to remove unwanted spaces
x2=cats.project_subject_categories.value_counts()

make_custom_plot2(title='Subject Category',total_counts=x,approvals_counts=x1,rejects_counts=x2,x_rotation_angle=80)
x= train.project_subject_subcategories.value_counts()
#prep for chart
x=x.sort_values(ascending=False)
x=x.iloc[0:20]

#chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[0])
plt.title("What are the frequent subject sub categories?",fontsize=16)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# Projects', fontsize=12)
plt.xlabel('Subject sub-Category', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
print("There are",len(train.project_subject_subcategories.unique()),"unique Subject Categories")
# Grouping similar categories for overall
subject_cats=','.join(train['project_subject_subcategories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_subcategories']) # to split on ","
cats.project_subject_subcategories=cats.project_subject_subcategories.str.strip() # to remove unwanted spaces
x=cats.project_subject_subcategories.value_counts()
print("There are ",len(x)," different subject sub-categories after Cleaning")
print("They are:- ",x.index.values)
# repeat for approved group and rejected group
# Grouping similar categories for approved 
subject_cats=','.join(approvals['project_subject_subcategories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_subcategories']) # to split on ","
cats.project_subject_subcategories=cats.project_subject_subcategories.str.strip() # to remove unwanted spaces
x1=cats.project_subject_subcategories.value_counts()

# Grouping similar categories for rejected
subject_cats=','.join(rejects['project_subject_subcategories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_subcategories']) # to split on ","
cats.project_subject_subcategories=cats.project_subject_subcategories.str.strip() # to remove unwanted spaces
x2=cats.project_subject_subcategories.value_counts()
# Plotting top 8 to avoid clutter
make_custom_plot2(title='Subject Sub-Category(Top-8)',total_counts=x.iloc[0:8],approvals_counts=x1.iloc[0:8],rejects_counts=x2.iloc[0:8],x_rotation_angle=80)
end_preprocess=time.time()
print("Time till sub-category:",end_preprocess-start_time,"s")
state_vals=train.groupby(['school_state'])['project_is_approved','teacher_id'].agg(
    {'project_is_approved':['sum','count'],'teacher_id':['nunique']}).reset_index()
state_vals.columns=['state','approved','total','teacher_count']
state_vals['approval_perc']=(state_vals.approved*100)/state_vals.total
state_vals['proj_per_teacher']=(state_vals.total)/state_vals.teacher_count
state_vals=state_vals.round(2)
print("Top States")
state_vals.sort_values('total',ascending=False).head()
import plotly.offline as py
py.init_notebook_mode(connected=True)


state_vals['text'] = 'Approved Projects: '+state_vals['approved'].astype(str)+ '<br>'+'Total Projects:'+state_vals['total'].astype(str)+'<br>'+\
    'Total Teachers:'+state_vals['teacher_count'].astype(str)+ '<br>'+'# projects per teacher:' +state_vals['proj_per_teacher'].astype(str)+'<br>'+\
    'Approval Rate: '+state_vals['approval_perc'].astype(str)
scl = [[0.0, 'rgb(0,39,143)'],[0.2, 'rgb(0,107,177)'],[0.4, 'rgb(0,154,200)'],\
            [0.6, 'rgb(0,189,220)'],[0.8, 'rgb(0,218,235)'],[1.0, 'rgb(0,240,247)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_vals['state'],
        z = state_vals['approval_perc'],
        locationmode = 'USA-states',
        text = state_vals['text'],
        marker = dict(
            line = dict (
                color = 'rgb(0,200,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Approval Percentage")
        ) ]

layout = dict(
        title = 'State wide Approval Analysis <br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False,filename='d3-cloropleth-map' )
approvals['project_submitted_datetime'] = pd.to_datetime(approvals['project_submitted_datetime'])
approvals['datetime_dow'] = approvals['project_submitted_datetime'].dt.dayofweek
approvals['datetime_year'] = approvals['project_submitted_datetime'].dt.year
approvals['datetime_month'] = approvals['project_submitted_datetime'].dt.month
approvals['datetime_hour'] = approvals['project_submitted_datetime'].dt.hour
approvals['datetime_day'] = approvals['project_submitted_datetime'].dt.day
approvals['datetime_date'] = approvals['project_submitted_datetime'].dt.date

rejects['project_submitted_datetime'] = pd.to_datetime(rejects['project_submitted_datetime'])
rejects['datetime_dow'] = rejects['project_submitted_datetime'].dt.dayofweek
rejects['datetime_year'] = rejects['project_submitted_datetime'].dt.year
rejects['datetime_month'] = rejects['project_submitted_datetime'].dt.month
rejects['datetime_hour'] = rejects['project_submitted_datetime'].dt.hour
rejects['datetime_day'] = rejects['project_submitted_datetime'].dt.day
rejects['datetime_date'] = rejects['project_submitted_datetime'].dt.date
app_date=approvals.groupby('datetime_date')['datetime_date'].count()
rej_date=rejects.groupby('datetime_date')['datetime_date'].count()
import plotly.plotly as py
from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go
# Make plotly work with Jupyter notebook
init_notebook_mode()
approved = go.Scatter(
    x=app_date.index,
    y=app_date.values,
    name = "# of Approved Projects",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

rejected = go.Scatter(
    x=rej_date.index,
    y=rej_date.values,
    name = "# of Rejected Projects",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8)

data = [approved,rejected]

layout = dict(
    title='Approved and Rejected projects over Time(with Rangeslider)',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=14,
                     label='2w',
                     step='day',
                     stepmode='backward'),
                dict(count=7,
                     label='1w',
                     step='day',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, validate=False,filename = "Approval with Rangeslider")
df=approvals.groupby(['datetime_day','datetime_month'])['project_is_approved'].sum()
df=df.reset_index()
df=df.pivot(index='datetime_day',columns='datetime_month')[['project_is_approved']]
df.columns = df.columns.droplevel()
df=df.reset_index()
# import calender
df.columns = ['x','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'] 
# Initialize the figure
plt.style.use('seaborn-darkgrid')
# create a color palette
palette = plt.get_cmap('Set1')
plt.figure(figsize=(15,10))

gridspec.GridSpec(4,3) 
plt.subplots_adjust(hspace=0.4)
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1
     # Find the right spot on the plot
    plt.subplot(4,3, num)
     # plot every groups, but discreet
    for v in df.drop('x', axis=1):
        plt.plot(df['x'], df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
     # Plot the lineplot
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=3.2, alpha=0.9, label=column)
     # Same limits for everybody!
    plt.xlim(0,32)
    plt.ylim(-2,2000)
     # Not ticks everywhere
    if num in range(10) :
        plt.tick_params(labelbottom='off')
    if num not in [1,4,7,10] :
        plt.tick_params(labelleft='off')
     # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )
# general title
plt.suptitle("# of Approvals - Variation Across Months", fontsize=20, fontweight=0, color='black', style='italic', y=1.02)
 
plt.show()
text_col='project_title'
target_col='project_is_approved'
# Borrowed from previous work at https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda
text_stats=pd.DataFrame()
text_stats[target_col]=train[target_col]
#Word count 
text_stats['word_count']=train[text_col].apply(lambda x: len(str(x).split()))
#Unique word 
text_stats['count_unique_word']=train[text_col].apply(lambda x: len(set(str(x).split())))
# text_stats.groupby('project_is_approved').mean()
temp=train.groupby('project_title')['project_is_approved'].agg(['sum','count'])
temp['approval_rate']=(temp['sum']*100)/temp['count']
temp.columns=['# of projects approved','# of total projects','Approval rate']
temp=temp.sort_values(by='# of total projects',ascending=False)
temp=temp.iloc[0:20]
temp
approved_mask=np.array(Image.open("../input/imagesfordonorchoose/1_approve_heart.png"))
rejected_mask=np.array(Image.open("../input/imagesfordonorchoose/2_reject_heart.png"))
# simple invert mask              # This is done to fill the words within the shape and not outside it
approved_mask=~approved_mask[:,:,1]
rejected_mask=~rejected_mask[:,:,1]
make_word_clouds('project_title',approved_mask,rejected_mask,"What does a good title contain?")
# Before performing changes , simple check
x=train[train.project_essay_3.notnull()]
print("The last time an entry occured in Project essay 3 -- ",x['project_submitted_datetime'].max())
# Making the First essay column :student_description
train['student_description']=train['project_essay_1']
#performing the adjustment
# df.loc[selection criteria, columns I want] = value
train.loc[train.project_essay_3.notnull(),'student_description']=train.loc[train.project_essay_3.notnull(),'project_essay_1']+train.loc[train.project_essay_3.notnull(),'project_essay_2']
#repeat for test dataset
test['student_description']=test['project_essay_1']
test.loc[test.project_essay_3.notnull(),'student_description']=test.loc[test.project_essay_3.notnull(),'project_essay_1']+test.loc[test.project_essay_3.notnull(),'project_essay_2']
# Making the second essay column : project_description
train['project_description']=train['project_essay_2']
#performing the adjustmen
# df.loc[selection criteria, columns I want] = value
train.loc[train.project_essay_3.notnull(),'project_description']=train.loc[train.project_essay_3.notnull(),'project_essay_3']+train.loc[train.project_essay_3.notnull(),'project_essay_4']
test['project_description']=test['project_essay_2']
test.loc[test.project_essay_3.notnull(),'project_description']=test.loc[test.project_essay_3.notnull(),'project_essay_3']+test.loc[test.project_essay_3.notnull(),'project_essay_4']
# check
test[test.project_essay_3.notnull()].head(1).project_description.values
#remove unwanted colunms
del(train['project_essay_1'])
del(train['project_essay_2'])
del(train['project_essay_3'])
del(train['project_essay_4'])
del(test['project_essay_1'])
del(test['project_essay_2'])
del(test['project_essay_3'])
del(test['project_essay_4'])

#update the subsets
approvals=train[train.project_is_approved==1]
rejects=train[train.project_is_approved==0]
get_text_stats(text_col='student_description')
rejected_mask=np.array(Image.open("../input/imagesfordonorchoose/1_approve_hand.png"))
approved_mask=np.array(Image.open("../input/imagesfordonorchoose/2_reject_hand.png"))
# simple invert mask
approved_mask=~approved_mask[:,:,1]
rejected_mask=~rejected_mask[:,:,1]
make_word_clouds('student_description',approved_mask,rejected_mask,"How are Students described in Approved VS Rejected projects?")
get_text_stats(text_col='project_description')
approved_mask=np.array(Image.open("../input/imagesfordonorchoose/1_approve_tick.png"))
rejected_mask=np.array(Image.open("../input/imagesfordonorchoose/2_reject_tick.png"))
# simple invert mask
approved_mask=~approved_mask[:,:,1]
rejected_mask=~rejected_mask[:,:,1]
make_word_clouds('project_description',approved_mask,rejected_mask,"How are Projects described in Approved VS Rejected projects?")
get_text_stats('project_resource_summary')
grade_mask_1=np.array(Image.open("../input/imagesfordonorchoose/grade1.png"))
grade_mask_2=np.array(Image.open("../input/imagesfordonorchoose/grade2.png"))
grade_mask_3=np.array(Image.open("../input/imagesfordonorchoose/grade3.png"))
grade_mask_4=np.array(Image.open("../input/imagesfordonorchoose/grade4.png"))

# simple invert mask
grade_mask_1=~grade_mask_1[:,:,1]
grade_mask_2=~grade_mask_2[:,:,1]
grade_mask_3=~grade_mask_3[:,:,1]
grade_mask_4=~grade_mask_4[:,:,1]

plt.figure(figsize=(16,12))
plt.suptitle("Do different grades request for different Items?", fontsize=20)
plt.subplot(221)
text=train[train.project_grade_category=='Grades PreK-2'].project_resource_summary
wc= WordCloud(background_color="white",max_words=7000,stopwords=stopword,mask=grade_mask_1,normalize_plurals=True)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Grade-PreK-2 Projects", fontsize=16)
#https://matplotlib.org/examples/color/colormaps_reference.html for colormaps
plt.imshow(wc.recolor(colormap='Dark2',random_state=17), alpha=0.98,interpolation='bilinear')

plt.subplot(222)
text=train[train.project_grade_category=='Grades 3-5'].project_resource_summary
wc= WordCloud(background_color="black",max_words=7000,stopwords=stopword,mask=grade_mask_2,normalize_plurals=True)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Grades 3-5 Projects", fontsize=16)
plt.imshow(wc.recolor(colormap='Pastel1',random_state=17), alpha=0.98,interpolation='bilinear')

plt.subplot(223)
text=train[train.project_grade_category=='Grades 6-8'].project_resource_summary
wc= WordCloud(background_color="black",max_words=7000,stopwords=stopword,mask=grade_mask_3,normalize_plurals=True)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Grades 6-8 Projects", fontsize=16)
plt.imshow(wc.recolor(random_state=17), alpha=0.98,interpolation='bilinear')

plt.subplot(224)
text=train[train.project_grade_category=='Grades 9-12'].project_resource_summary
wc= WordCloud(background_color="white",max_words=7000,stopwords=stopword,mask=grade_mask_4,normalize_plurals=True)
wc.generate(" ".join(text))
plt.axis("off")
plt.title("Words frequented in Grades 9-12 Projects", fontsize=16)
#https://matplotlib.org/examples/color/colormaps_reference.html for colormaps
plt.imshow(wc.recolor(colormap='Dark2',random_state=17), alpha=0.98,interpolation='bilinear')
plt.show()
end_time=time.time()
print("Time till plotting section",end_time-start_time,"s")
resources['total_cost']=resources['quantity']*resources['price']
# Group by and get concat of the description
resources['description']=resources['description'].astype(str)
x=resources.groupby('id')['description'].apply(lambda x: "%s" % ', '.join(x))   #https://stackoverflow.com/questions/17841149/pandas-groupby-how-to-get-a-union-of-strings
x.head(2)
# project level resource stats
resources_agg=resources.groupby('id')['quantity','price','total_cost'].agg({'quantity':['sum','count'],'price':['mean'],'total_cost':['sum']})
resources_agg.columns=['item_quantity_sum','variety_of_items','avg_price_per_item','total_cost']
resources_agg['collated_description']=x
resources_agg=resources_agg.reset_index()
#resources_agg=resources_agg.sort_values("total_cost",ascending=False)
train_merge=pd.merge(left=train,right=resources_agg,on='id',how='left')
train_merge.sort_values("total_cost",ascending=False).head()
all_data=pd.concat([train,test],axis=0)
all_data_merge=pd.merge(left=all_data,right=resources_agg,on='id',how='left')
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

def preprocess_and_clean(text_col):
    word_list = gensim.utils.simple_preprocess(text_col, deacc=True)
    clean_words = [w for w in word_list if not w in eng_stopwords]
    clean_words=[lem.lemmatize(word, "v") for word in clean_words]
    return(clean_words)
x=all_data_merge['collated_description'].apply(preprocess_and_clean)
bigram_transformer = gensim.models.Phrases(x)
x=x.apply(lambda word : bigram_transformer[word])
#take a peak to check
x.head()
model = Word2Vec(x, size=100, window=5, min_count=10, workers=4,seed=10)
from sklearn.manifold import TSNE

def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points')
    plt.xlim(x_coords.min()+10, x_coords.max()+10)
    plt.ylim(y_coords.min()+10, y_coords.max()+10)
    plt.show()
word='apple'
print("Word Embedding of words similar to the word:- ",str(word))
display_closestwords_tsnescatterplot(model,word)
word='chair'
print("Word Embedding of words similar to the word:- ",str(word))
display_closestwords_tsnescatterplot(model,word)
word='shirt'
print("Word Embedding of words similar to the word:- ",str(word))
display_closestwords_tsnescatterplot(model,word)
word='art'
print("Word Embedding of words similar to the word:- ",str(word))
display_closestwords_tsnescatterplot(model,word)
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

#create the dictionary
dictionary = Dictionary(x)
print("There are",len(dictionary),"number of words in the final dictionary")
corpus = [dictionary.doc2bow(text) for text in x]
#create the Lda model
ldamodel = LdaModel(corpus=corpus, num_topics=15, id2word=dictionary)
end_lda=time.time()
print("Time till Lda model creation:",end_lda-start_time,"s")
import pyLDAvis.gensim

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
#creating the topic probability matrix 
topic_probability_mat = ldamodel[corpus]
#split it to test and train
train_matrix=topic_probability_mat[:train.shape[0]]
test_matrix=topic_probability_mat[train.shape[0]:]

# check one entry
x.iloc[6]
start=time.time()
doc = x.iloc[6]
vec_bow = dictionary.doc2bow(doc)
vec_lda = ldamodel[vec_bow] # convert the query to Lda space
print(vec_lda)
end=time.time()
print("Time for one iter:",end-start,"s")
from gensim import corpora, models, similarities
index = similarities.MatrixSimilarity(ldamodel[corpus]) 
sims = index[vec_lda] 
sims = sorted(enumerate(sims), key=lambda item: -item[1])


end_lda=time.time()
print("Time till Lda model viz:",end_lda-start_time,"s")
del(ldamodel)
del(corpus)
del(dictionary)
del(model)
gc.collect()
# To be done
x=train_merge.total_cost.value_counts()
#sort by price
x=x.sort_index(ascending=False)
#subset for alteast 5 entries
x=x[x>5]
# get the top 20
x=x.iloc[0:20]
#plot
plt.figure(figsize=(16,5))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Frequent(more than 5 projects) Price points")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Price point', fontsize=12)
plt.show()
# subset expedition kit requests
subset=train_merge[train_merge.total_cost==9999]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==0].head()
# subset expedition kit requests
subset=train_merge[train_merge.total_cost==6999]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==0].head()
# subset expedition kit requests
subset=train_merge[train_merge.total_cost==3999]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==1].head()
# subset expedition kit requests
subset=train_merge[train_merge.total_cost==4995.95]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==1].head()

price_points=[1999.99, 1999.96, 1999.95,1999.9]
# subset expedition kit requests
subset=train_merge[train_merge.total_cost.isin(price_points)]
print("Number of times requested:",len(subset))
print("Number of times approved:",len(subset[subset.project_is_approved==1]))
subset[subset.project_is_approved==0].head()
# Using cleaning functions from previous work --> https://www.kaggle.com/jagangupta/understanding-the-topic-of-toxicity/notebook
def preprocess_and_clean(text_col):
    """
    Function to build tokenized texts from input comment and the clean them
    Following transformations will be done
    1) Stop words removal from the nltk stopword list
   #commenting out for speed issues 2) Bigram collation (Finding common bigrams and grouping them together using gensim.models.phrases) (Eg: new + york --> new_york )
    3) Lemmatization (Converting word to its root form : babies --> baby ; children --> child)
    """
    
    word_list = gensim.utils.simple_preprocess(text_col, deacc=True)
    #Phrases help us group together bigrams :  new + york --> new_york
    #bigram = gensim.models.Phrases(text_col)
    
    #remove stop words
    clean_words = [w for w in word_list if not w in eng_stopwords]
    #collect bigrams
    #clean_words = bigram[clean_words]
    #Lemmatize
    clean_words=[lem.lemmatize(word, "v") for word in clean_words]
    return(' '.join(clean_words))  

#check clean function
print("Before clean:",train.project_description.iloc[16])
print("After clean:",preprocess_and_clean(train.project_description.iloc[16]))
# Null treatment
train.teacher_prefix=train.teacher_prefix.fillna('Unknown')
test.teacher_prefix=test.teacher_prefix.fillna('Unknown')
train.gender=train.gender.fillna('Unknown')
test.gender=test.gender.fillna('Unknown')

print(train.shape)
print(test.shape)
y=train['project_is_approved']
train_id=train['id']
test_id=test['id']
del(train['project_is_approved'])
all_data=pd.concat([train,test],axis=0)
print(all_data.shape)
all_data_merge=pd.merge(left=all_data,right=resources_agg,on='id',how='left')
# taking some FE ideas from public kernals
# thanks owl, --> https://www.kaggle.com/the1owl/the-choice-is-yours
# and jmbull --> https://www.kaggle.com/jmbull/xtra-credit-xgb-w-tfidf-feature-stacking
from sklearn import *
from tqdm import tqdm
# Label encode some columns
cols = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category',
    'project_subject_categories', 
    'project_subject_subcategories',
    'gender']
for c in tqdm(cols):
    le = preprocessing.LabelEncoder()
    le.fit(all_data_merge[c].astype(str))
    all_data_merge[c] = le.transform(all_data_merge[c].astype(str))
# Log1p transform price columns
all_data_merge['avg_price_per_item']=np.log1p(all_data_merge['avg_price_per_item'])
all_data_merge['total_cost']=np.log1p(all_data_merge['total_cost'])

# date features
all_data_merge['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
all_data_merge['datetime_dow'] = all_data_merge['project_submitted_datetime'].dt.dayofweek
all_data_merge['datetime_year'] = all_data_merge['project_submitted_datetime'].dt.year
all_data_merge['datetime_month'] = all_data_merge['project_submitted_datetime'].dt.month
all_data_merge['datetime_hour'] = all_data_merge['project_submitted_datetime'].dt.hour
all_data_merge['datetime_day'] = all_data_merge['project_submitted_datetime'].dt.day
#process text cols
text_cols=['project_title', 
           'collated_description',
           'project_resource_summary',
           'student_description', 
           'project_description']
for c in tqdm(text_cols):
    all_data_merge[c+'_len']=all_data_merge[c].apply(len)               # get length (ie) letter count
    all_data_merge[c+'_word_count']=all_data_merge[c].apply(lambda x: len(str(x).split())) # get word count
    all_data_merge[c]=all_data_merge[c].apply(preprocess_and_clean)
end_time=time.time()
print("Time till end",end_time-start_time,"s")
train_shape=train.shape
test_shape=test.shape
# del(train)
# del(test)
del(train_merge)
del(resources)
del(resources_agg)
del(subset)
del(approvals)
del(rejects)
del(all_data)
gc.collect()
############################
# needs debugging
########################
# from sklearn.pipeline import FeatureUnion,TransformerMixin,Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.metrics import auc

# # https://www.kaggle.com/lopuhin/eli5-for-mercari
# # https://github.com/scikit-learn/scikit-learn/issues/2034

# class GetItemTransformer(TransformerMixin):
#     """
#     Custom class to fetch just the column needed from the numpy nd array from pandas.values that is passed to the vectorizer
#     """
#     def __init__(self, field):
#         self.field = field
#     def fit(self, X, y=None):
#         return self
#     def transform(self,X):
#         field_idx = list(all_data_merge.columns).index(self.field)
#         return X[:,field_idx]
    

# vectorizer = FeatureUnion([
#     ('project_title',
#          Pipeline([
#             ('get', GetItemTransformer('project_title')),
#             ('vectorize',CountVectorizer(
#             ngram_range=(1, 2),
#             max_features=2000))
#          ])),
#     ('project_resource_summary',
#          Pipeline([
#             ('get', GetItemTransformer('project_resource_summary')) ,
#             ('vectorize',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=5000))
#          ])),
#     ('student_description',
#          Pipeline([
#             ('get', GetItemTransformer('student_description')) ,
#             ('vectorize',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=20000))
#          ])),
#     ('project_description',
#          Pipeline([
#             ('get', GetItemTransformer('project_description')),
#             ('vectorize',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=20000))
#          ])),
#     ('collated_description',                                                 # using count vect here as this coulmn contains mostly product descr. 
#          Pipeline([
#             ('get', GetItemTransformer('collated_description')),
#             ('vectorize',CountVectorizer(
#             ngram_range=(1, 2),
#             max_features=10000))
#          ]))
# ])
# error due to pipeline not having get feature names!! --> https://github.com/scikit-learn/scikit-learn/issues/6424
# # todo : to add other fields(non-text) into the pipeline itself
del(all_data_merge['project_submitted_datetime'])
#https://stackoverflow.com/questions/29815129/pandas-dataframe-to-list-of-dictionaries
all_data_merge_1=all_data_merge.to_dict('records')
del(all_data_merge)
gc.collect()
# from sklearn.pipeline import FeatureUnion,TransformerMixin,Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.metrics import auc
# import dill as pickle
# # https://www.kaggle.com/lopuhin/eli5-for-mercari
# # https://github.com/scikit-learn/scikit-learn/issues/2034

# vectorizer = FeatureUnion([
#         ('project_title',CountVectorizer(
#             ngram_range=(1, 2),
#             max_features=5000,
#             preprocessor=lambda x: x['project_title'])),
#         ('project_resource_summary',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=10000,
#             preprocessor=lambda x: x['project_resource_summary'])),
#         ('student_description',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=30000,
#             preprocessor=lambda x: x['student_description'])),
#         ('project_description',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=30000,
#             preprocessor=lambda x: x['project_description'])),
#         ('char_project_resource_summary',TfidfVectorizer(
#             ngram_range=(3,5),analyzer='char',
#             max_features=5000,
#             preprocessor=lambda x: x['project_resource_summary'])),
#         ('char_student_description',TfidfVectorizer(
#             ngram_range=(3,5),analyzer='char',
#             max_features=5000,
#             preprocessor=lambda x: x['student_description'])),
#         ('char_project_description',TfidfVectorizer(
#             ngram_range=(3,5),analyzer='char',
#             max_features=5000,
#             preprocessor=lambda x: x['project_description'])),   
    
#         ('collated_description',CountVectorizer(
#             ngram_range=(1, 2),
#             max_features=30000,
#             preprocessor=lambda x: x['collated_description'])),
#         ('Non_text',DictVectorizer())
#     ],njobs=4)

## Parallel processing didn't work 
from sklearn.pipeline import FeatureUnion,TransformerMixin,Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import auc
# import dill as pickle
# https://www.kaggle.com/lopuhin/eli5-for-mercari
# https://github.com/scikit-learn/scikit-learn/issues/2034
def get_col(col_name):
    return lambda x: x[col_name]

vectorizer = FeatureUnion([
        ('project_title',CountVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            preprocessor=get_col('project_title'))),
        ('project_resource_summary',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            preprocessor=get_col('project_resource_summary'))),
        ('student_description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100000,
            preprocessor=get_col('student_description'))),
        ('project_description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100000,
            preprocessor=get_col('project_description'))),
        ('collated_description',CountVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            preprocessor=get_col('collated_description'))),
        ('Non_text',DictVectorizer())
    ])
start_vect=time.time()
all_data_vectorized = vectorizer.fit_transform(all_data_merge_1)
# split train and test 
# train_text_data_vectorizer=vectorizer.fit_transform(all_data_merge.iloc[:train_shape[0]])
# test_text_data_vectorizer=vectorizer.fit_transform(all_data_merge.iloc[train_shape[0]:])

end_time=time.time()

print("total time in vectorization creation",end_time-start_vect,"s")
print("total time till vectorization creation",end_time-start_time,"s")
from scipy.sparse import csr_matrix, hstack
final_dataset=all_data_vectorized.tocsr()
# older version
# all_text_data_vectorized = vectorizer.fit_transform(all_data_merge.values)
# train_x_only_text=all_text_data_vectorized[0:train_shape[0]]
# test_x_only_text=all_text_data_vectorized[train_shape[0]:]
# all_cols=all_data_merge.columns.values
# non_text_cols = [col for col in all_cols if col not in text_cols]
# non_text_cols.remove('id')
# non_text_cols.remove('project_submitted_datetime')
# from scipy.sparse import csr_matrix, hstack
# all_non_text_data = csr_matrix(all_data_merge[non_text_cols].values)
# final_dataset=hstack((all_text_data_vectorized,all_non_text_data)).tocsr()
# end_time=time.time()
# print("total time till Sparse mat creation",end_time-start_time,"s")
train_x=final_dataset[0:train_shape[0]]
test_x=final_dataset[train_shape[0]:]
del final_dataset,all_data_vectorized
gc.collect()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_x, y, test_size=0.33, random_state=2018)
# Using LGBM params from https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code
params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16,
        'num_leaves': 31,
        'learning_rate': 0.25,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':1234
}  
import lightgbm as lgb

model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose_eval=25)
from sklearn.metrics import roc_auc_score
valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
test_preds = model.predict(test_x, num_iteration=model.best_iteration)
auc = roc_auc_score(y_valid, valid_preds)
print('AUC:',auc)
plt.show()
end_time=time.time()
print("total time till LGB model",end_time-start_time,"s")
import xgboost as xgb
xgb_params = {'eta': 0.2, 
                  'max_depth': 5, 
                  'subsample': 0.8, 
                  'colsample_bytree': 0.8, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': 1234
                 }
# d_train = xgb.DMatrix(X_train, y_train)
# d_valid = xgb.DMatrix(X_valid, y_valid)
# d_test = xgb.DMatrix(test_x)
X_train, X_valid, y_train, y_valid = train_test_split(train_x, y, test_size=0.33, random_state=2018)
#for eli5
d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_valid, y_valid)
d_test = xgb.DMatrix(test_x)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model_xgb = xgb.train(xgb_params, d_train, 500, watchlist, verbose_eval=50, early_stopping_rounds=20)
xgb_pred_test = model_xgb.predict(d_test)
xgb_pred_valid = model_xgb.predict(d_valid)
auc = roc_auc_score(y_valid, xgb_pred_valid)
print('AUC:',auc)
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_valid, xgb_pred_valid)
roc_auc = metrics.auc(fpr, tpr)
fpr_1,tpr_1,thresholds_1=roc_curve(y_valid,valid_preds)
roc_auc_1 = metrics.auc(fpr_1, tpr_1)
plt.figure(figsize=(8,6))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'XGBoost-AUC = %0.2f' % roc_auc)
plt.plot(fpr_1, tpr_1, 'g', label = 'LGBM-AUC = %0.2f' % roc_auc_1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# end_time=time.time()
# print("total time till XBG model",end_time-start_time,"s")
xgb_pred_train = model_xgb.predict(d_train)
import eli5
# eli5.explain_weights_lgb(model_xgb, vec=vectorizer)     # out of bounds error
# text_features=vectorizer.get_feature_names()
# other_features=non_text_cols
# all_features=text_features+non_text_cols
# eli5.explain_weights_xgboost(model_xgb, feature_names=all_features)          
eli5.show_weights(model_xgb,vec=vectorizer)    
# random entry
print("Project is Approved?:Actual",y[100])
print("Project is Approved?:Predicted prob:",xgb_pred_train[100])
display(eli5.show_prediction(model_xgb, doc=all_data_merge_1[100], vec=vectorizer,show_feature_values=True,top=20))    
# random entry
print("Project is Approved?:Actual",y[500])
print("Project is Approved?:Predicted prob:",xgb_pred_train[500])
display(eli5.show_prediction(model_xgb, doc=all_data_merge_1[500], vec=vectorizer,show_feature_values=True,top=20))    
from IPython.display import display
no_missing = lambda feature_name, feature_value: not np.isnan(feature_value)
for i in range(5):
    print("Project is Approved?:Actual",y[i])
    print("Project is Approved?:Predicted prob:",xgb_pred_train[i])
    display(eli5.show_prediction(model_xgb, doc=all_data_merge_1[i], vec=vectorizer,show_feature_values=True,top=30,feature_filter=no_missing))  
final_preds=0.4*xgb_pred_test+0.6*test_preds
# Making submission
x_preds = pd.DataFrame(final_preds)
x_preds.columns = ['project_is_approved']
sub_id=sample_sub['id']
submission = pd.concat([sub_id, x_preds], axis=1)
submission.to_csv('lgbm_xgb_blend.csv', index=False)
end_time=time.time()
print("total time spent in FE and model",end_time-start_vect,"s")
print("total time till end",end_time-start_time,"s")
# To be continued....