from os import write
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
fig=plt.figure()
'''
df = pd.read_excel('./test1.xlsx')
writer=pd.ExcelWriter('./result_merged.xlsx')
industry_id=df['industry_id2']

ads=df['ad']
flows=df['flow']
y1=df['y1']
y2=df['y2']
y3=df['y3']
last=0
rows=-1
data=pd.DataFrame()
for industry in industry_id:
    rows+=1
    if industry!=last:
        if last!=0:
            plt.clf()
            heatmap=sns.heatmap(temp.values,annot=True,xticklabels=False,yticklabels=False)
            s=heatmap.get_figure()
            s.savefig(str(last)+'.jpg')
            #plt.show()
            temp=pd.concat([temp,pd.DataFrame(index=['industry_id:'+str(industry)],columns=range(1,5))])
            data=pd.concat([data,temp])
        last=industry
        temp=pd.DataFrame(np.zeros([5,4]),index=range(1,6),columns=range(1,5))
    temp.loc[ads[rows],flows[rows]]=y1[rows]

print(data)
data.to_excel(excel_writer=writer)
writer.save()
'''

'''
df = pd.read_csv('./data.csv')
print(df)
for i in range(len(df)):
    if df.loc[i,'industry_id2']!=-1:
        df.loc[i,'industry_id2']=int(str(df.loc[i,'industry_id2'])[0:-2])
print(df)
df.to_csv('./data1.csv')
'''

df = pd.read_excel('./test.xlsx')
writer=pd.ExcelWriter('./result1.xlsx')
industry_id=df['industry_id2']

ads=df['ad']
flows=df['flow']
y1=df['y1']
y2=df['y2']
y3=df['y3']
last=0
rows=-1
data=pd.DataFrame()
for industry in industry_id:
    rows+=1
    if industry!=last:
        if last!=0:
            plt.clf()
            temp=pd.concat([temp,pd.DataFrame(index=['industry_id:'+str(industry)],columns=range(1,5))])
            data=pd.concat([data,temp])
            if temp.loc[3,2]+temp.loc[2,3]-temp.loc[2,2]-temp.loc[3,3]>0:
                print(last)
        last=industry
        temp=pd.DataFrame(np.zeros([5,4]),index=range(1,6),columns=range(1,5))
    temp.loc[ads[rows],flows[rows]]=y1[rows]
    