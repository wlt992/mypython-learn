import pandas as pd
import requests 
import time
from selenium import webdriver 
import numpy as np

driver = webdriver.Firefox() #初始化浏览器，用于抓取动态数字

#初始化抓取的URL列表
second_hand_page_url = pd.DataFrame(columns = ["url"])

#初始化抓取的房源信息列表
second_hand_page_infor = pd.DataFrame(columns = ["楼层",  "位置", "年代",
                                                 "房型", "面积", "朝向",
                                                 "楼层", "装修程度", "房屋单价", 
                                                 "参考首付", "参考月供", "总价", "url"])



#得到每个url对应的的源代码，便于后续解析出想要的信息
def get_page_code(url):
    #模拟一个设备，否则网站限制访问
    head = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.118 Safari/537.36'}
    html = requests.get(url, headers=head)
    html.encoding = 'utf-8'  #转换成中文编码，否则可能会出现乱码
    
    return html.text

#得到列表页中每条房源信息对应的url，存放至建立好的dataframe中，为后续爬取详细信息做准备
def get_page_url(page_code):
    index1 = page_code.find('<div class="house-title">')

    while index1 != -1 :
        page_code = page_code[index1 + 25:]
        house_url = page_code[page_code.find('href="') + 6:page_code.find('" target=\'_blank\' class="houseListTitle ">')]
        second_hand_page_url.loc[second_hand_page_url.shape[0]] = house_url
        index1 = page_code.find('<div class="house-title">')
    return 1

#得到列表页中下一页对应的url，便于循环爬取所有的房源信息页面
def get_next_url(page_code):
    if page_code.find('" class="aNxt">下一页 &gt;</a>') == -1:
        return "none"
    page_code = page_code[page_code.find('<div class="multi-page">'):page_code.find('" class="aNxt">下一页 &gt;</a>')]
    times = page_code.count('<i class="aDotted">...</i>')
    index = findstr(page_code, '<i class="aDotted">...</i>', times)
    if index != -1 :
        page_code = page_code[index:]

    times = page_code.count('<a href="')
    index = findstr(page_code, '<a href="', times)
    
    if index != -1:
        page_code = page_code[index + 9:]
        return page_code
    
    return "none"
   

    
#得到网页中房源的相关信息
def get_page_infor(url):
    page_code = get_page_code(url)
    infor = np.repeat('', 13).tolist()
    infor[0] = page_code[page_code.find("<dl><dt>小区：</dt>"):page_code.find("<dl><dt>位置：</dt>")]
    infor[0] = infor[0][infor[0].find("propview>") + 9:infor[0].find("</a>")]

    infor[1] = page_code[page_code.find("<dl><dt>位置：</dt>"):page_code.find("<dl><dt>年代：</dt>")]
    temp_address = infor[1][infor[1].find("propview>") + 9:infor[1].find("</a>")]
    temp_infor = infor[1][infor[1].find("</a>") + 4:]
    temp_address = temp_address + "－" + temp_infor[temp_infor.find("propview>") + 9:temp_infor.find("</a>")]
    temp_infor = temp_infor[temp_infor.find("propview>"):]
    temp_address = temp_address + replace_none(temp_infor[temp_infor.find("</a>") + 4:temp_infor.find("</p>")])
    infor[1] = temp_address

    infor[2] = page_code[page_code.find("<dl><dt>年代：</dt>"):page_code.find("<dl><dt>类型：</dt>")]
    infor[2] = infor[2][infor[2].find("<dd>") + 4:infor[2].find("</dd>")]

    infor[3] = page_code[page_code.find("<dl><dt>房型：</dt>"):page_code.find("<dl><dt>面积：</dt>")]
    infor[3] = infor[3][infor[3].find("<dd>") + 4:infor[3].find("</dd>")]
    infor[3] = replace_none(infor[3])

    infor[4] = page_code[page_code.find("<dl><dt>面积：</dt>"):page_code.find("<dl><dt>朝向：</dt>")]
    infor[4] = infor[4][infor[4].find("<dd>") + 4:infor[4].find("平方米")] + "平方米"
    
    infor[5] = page_code[page_code.find("<dl><dt>朝向：</dt>"):page_code.find("<dl><dt>楼层：</dt>")]
    infor[5] = infor[5][infor[5].find("<dd>") + 4:infor[5].find("</dd>")]

    infor[6] = page_code[page_code.find("<dl><dt>楼层：</dt>"):page_code.find("<dl><dt>装修程度：</dt>")]
    infor[6] = infor[6][infor[6].find("<dd>") + 4:infor[6].find("</dd>")]
    infor[6] = replace_none(infor[6])
    
    infor[7] = page_code[page_code.find("<dl><dt>装修程度：</dt>"):page_code.find("<dl><dt>房屋单价：</dt>")]
    infor[7] = infor[7][infor[7].find("<dd>") + 4:infor[7].find("</dd>")]
    

    infor[8] = page_code[page_code.find("<dl><dt>房屋单价：</dt>"):page_code.find("<dl><dt>参考首付：</dt><")]
    infor[8] = infor[8][infor[8].find("<dd>") + 4:infor[8].find("</dd>")]
    
    infor[9] = page_code[page_code.find("<dl><dt>参考首付：</dt>"):page_code.find("<dl><dt>参考月供：</dt>")]
    infor[9] = infor[9][infor[9].find("<dd>") + 4:infor[9].find("</dd>")].strip()
    
    infor[10] = page_code[page_code.find("<dl><dt>参考月供：</dt>"):page_code.find("<span id=\"reference_monthpay\">")]
    infor[10] = infor[10][infor[10].find("<dd>") + 4:infor[10].find("</dd>")]
    driver.get(url)
    infor[10] = driver.find_element_by_id("reference_monthpay").text


    infor[11] = page_code[page_code.find('<span class="light info-tag">'):page_code.find('<span class="info-tag"><em>')]
    infor[11] = infor[11][infor[11].find("<em>") + 4:infor[11].find("</em>")] + infor[11][infor[11].find("</em>") + 5:infor[11].find("</span>")]
    
    infor[12] = url   
    
    
    return infor
    
def get_total_infor():
    global second_hand_page_infor
    infor = np.repeat('', 13).tolist()
    
    for index in range(second_hand_page_url.shape[0]) :
        print(index)
        time.sleep( 2 )
        url = second_hand_page_url.iloc[index][0]
        infor = get_page_infor(url)
        insertRow = pd.DataFrame([infor], columns=second_hand_page_infor.columns)
        second_hand_page_infor = second_hand_page_infor.append(insertRow, ignore_index=True)
    
    return 1
    
    
# 查找substr第times出现的位置
def findstr(str, substr, times):
    liststr = str.split(substr)
    if len(liststr) <= times:
        return -1
    return len(str) - len(liststr[-1]) - len(substr)
    
    
#替换空格和回车符
def replace_none(str):
    str=str.replace(" ","")
    str=str.replace("\n","")

    return  str



url = 'https://shanghai.anjuke.com/sale/changning/'
page_code = get_page_code(url)
get_page_url(page_code)
url = get_next_url(page_code)
while url != 'none' :
    print(url)
    time.sleep(10)  #睡眠10秒钟
    page_code = get_page_code(url)
    get_page_url(page_code)
    url = get_next_url(page_code)
      
get_total_infor()