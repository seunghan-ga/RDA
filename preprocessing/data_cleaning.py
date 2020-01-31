import pandas as pd
import pymysql

# 데이터 불러오기

db_info = {
    "host": "192.168.1.101",
    "user": "hadoop",
    "password": "hadoop",
    "db": "DEMO",
    "charset": "utf8"
}
conn = pymysql.connect(**db_info)
df = pd.read_sql_query("select * from sar", conn)

# 데이터 살펴보기
print(df.columns)
print(df.describe().transpose())

# 데이터 클리닝 1
# null 컬럼이 있는지 확인
print(df.isnull().any().any(), df.shape)

if df.isnull().any().any() == True:
    rows = df.shape[0]
    # null 컬럼 제거
    df = df.dropna()
    print(rows)
    print(df.isnull().any().any(), df.shape)

# 데이터 특징 상관 관계 파악
print(df[df.columns[4]].corr(df[df.columns[7]]))

for f in range(len(df.columns)):
    related = df['idel'].corr(df[df.columns[f+2]])
    print("%s: %s" % (df.columns[f+2], related))

# 데이터 클리닝 2
# 푸리에 변환을 이용하여 훈련용 데이터셋은 정상치만 있는지 확인을 한다.
train_fft = np.fft.fft(train)
test_fft = np.fft.fft(test)

# 훈련용 데이터 셋의 정상치만 있는것을 확인할 수 있는 차트
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train_fft[:,0].real, label='Bearing 1', color='blue', animated = True, linewidth=1)
ax.plot(train_fft[:,1].imag, label='Bearing 2', color='red', animated = True, linewidth=1)
ax.plot(train_fft[:,2].real, label='Bearing 3', color='green', animated = True, linewidth=1)
ax.plot(train_fft[:,3].real, label='Bearing 4', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Training Frequency Data', fontsize=16)
plt.show()


# test 영역 데이터에 불량 부분이 보임
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(test_fft[:,0].real, label='Bearing 1', color='blue', animated = True, linewidth=1)
ax.plot(test_fft[:,1].imag, label='Bearing 2', color='red', animated = True, linewidth=1)
ax.plot(test_fft[:,2].real, label='Bearing 3', color='green', animated = True, linewidth=1)
ax.plot(test_fft[:,3].real, label='Bearing 4', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Test Frequency Data', fontsize=16)
plt.show()

# 데이터 시각화

# 데이터 모델링
