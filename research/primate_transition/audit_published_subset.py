import pandas as pd

BASE='https://raw.githubusercontent.com/Sferrigno/RecursiveSequenceGeneration/master/'

def classify(seq):
    s=str(seq).strip().upper() if pd.notna(seq) else ''
    if s in {'ACDB','CABD'}: return 'CE'
    if s in {'ACBD','CADB'}: return 'Cross'
    if s in {'ABCD','CDAB'}: return 'Tail'
    return 'Other'

for name,file in [('ADULT','RecursionUSAdults.csv'),('KIDS','RecursionKids.csv')]:
    d=pd.read_csv(BASE+file)
    d=d[d['Order pressed'].notna()].copy()
    d['cls']=d['Order pressed'].map(classify)
    print('\n===',name,'TOTAL',len(d),'SUBJECTS',d['Sub'].nunique(),'===')
    cols=['List','Known','Session','NumLists'] if name=='ADULT' else ['List','EXCLUDE','Test or Training','NumLists']
    for c in cols:
        if c in d.columns:
            print('\n--',c,'--')
            print(d.groupby(c,dropna=False).agg(n=('cls','size'),CE=('cls',lambda x:(x=='CE').sum()),Cross=('cls',lambda x:(x=='Cross').sum()),subs=('Sub','nunique')).to_string())
    print('\n-- SUBJECT COUNTS --')
    print(d.groupby('Sub').agg(n=('cls','size'),CE=('cls',lambda x:(x=='CE').sum()),Cross=('cls',lambda x:(x=='Cross').sum()),lists=('List','nunique')).to_string())
    if name=='KIDS' and 'EXCLUDE' in d.columns:
        for v in sorted(d['EXCLUDE'].dropna().unique()):
            z=d[d['EXCLUDE']==v]
            print('EXCLUDE',v,'n',len(z),'subjects',z['Sub'].nunique(),'CE',int((z.cls=='CE').sum()),'Cross',int((z.cls=='Cross').sum()))
    if name=='ADULT':
        for listv,z in d.groupby('List'):
            print('LIST',listv,'n',len(z),'subjects',z.Sub.nunique(),'CE',int((z.cls=='CE').sum()),'Cross',int((z.cls=='Cross').sum()))
