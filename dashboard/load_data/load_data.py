from sqlalchemy import create_engine
import pandas as pd

#postgres data
UNAME = "postgres"
PWD = "9876"
HOST = "my_postgres"
PORT = "5432"
DB = 'postgres'

#connection to postgres container
engine = create_engine(f"postgresql://{UNAME}:{PWD}@{HOST}:{PORT}/{DB}", echo = True) 

df = pd.read_csv('./df_world.csv')
df.to_sql('overshootday', engine, if_exists='replace', index=False)

df_01 = pd.read_csv('./df_01.csv')
df_01.to_sql('sce01', engine, if_exists='replace', index=False)

df_02 = pd.read_csv('./df_02.csv')
df_02.to_sql('sce02', engine, if_exists='replace', index=False)

df_03 = pd.read_csv('./df_03.csv')
df_03.to_sql('sce02', engine, if_exists='replace', index=False)

df_04 = pd.read_csv('./df_04.csv')
df_04.to_sql('sce04', engine, if_exists='replace', index=False)

df_05 = pd.read_csv('./df_05.csv')
df_05.to_sql('sce05', engine, if_exists='replace', index=False)

#merge all df together
A = pd.merge(df,df_01, how='outer')
B = pd.merge(A,df_02, how='outer')
C = pd.merge(B,df_03, how='outer')
D = pd.merge(C,df_04, how='outer')
E = pd.merge(D,df_05, how='outer')
E.to_sql('overshootday_all', engine, if_exists='replace', index=False)

change_landtypes = {'cropLand': 117.00111482720175,
 'grazingLand': 18.265895953757227,
 'forestLand': 34.50980392156864,
 'fishingGround': 66.74816625916871,
 'builtupLand': 296.6666666666667,
 'carbon': 142.108251324754}

pd.DataFrame(change_landtypes, index=range(5)).to_sql('landtypes', engine, if_exists='replace', index=False)


