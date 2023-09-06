class Environment:
    PRICE_IDX = 4 # 종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data = chart_data    # 객체 내의 chart_data에 주어진 chart_data를 저장   # 2차원 배열, data frame 형태
        self.observation = None         # 차트 데이터에서 현재 위치의 관측 값
        self.idx = -1                   # 현재 위치
    
    # 차트 데이터의 처음으로 돌아감
    def reset(self): 
        self.observation = None
        self.idx = -1

    # 하루 앞으로 이동하며 차트 데이터에서 관측 데이터(observation)를 제공
    def observe(self):  
        if len(self.chart_data) > self.idx + 1: # 차트 데이터의 전체 길이보다 다음 위치가 작을 경우 가져올 데이터가 있다는 뜻
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx] # 차트 데이터에서 현재 위치의 요소 (행)을 가져옴 (iloc: index location)
            return self.observation
        return None     # 더 이상 제공할 데이터가 없으면 None 반환

    # 관측 데이터에서 종가를 반환. 종가 close가 5번째 열이기 때문에 PRICE_IDX = 4
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX] # observation은 하나의 행이고 여기서 인덱스 4값인 observation[4]가 종가에 해당.
        return None

    """ 삭제된 부분
    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
    """