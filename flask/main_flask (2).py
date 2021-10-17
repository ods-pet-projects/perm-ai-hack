from flask import Flask, request, Response, jsonify, flash, jsonify, redirect, url_for, send_from_directory, render_template

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/dsvm/perm/flask'

import html

import requests
import json

# open data libs
from dadata import Dadata


# parsing libs
import requests
from bs4 import BeautifulSoup
import re

from tika import parser

import re
import requests
import pandas as pd
import os

from bs4 import BeautifulSoup
import requests
from requests import Request, Session
import pandas as pd
import re
from ftfy import fix_text

#############################################################################

import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
import pymorphy2
from pymystem3 import Mystem
import xml.etree.ElementTree as ET
from deeppavlov.models.tokenizers.ru_tokenizer import RussianTokenizer
from stop_words import get_stop_words
from string import punctuation
from ftfy import fix_text



import requests
import json
import re
import pandas as pd
import numpy as np

from ftfy import fix_text
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.nn import Softmax 
import torch

tokenizer = AutoTokenizer.from_pretrained("../rubert-toxic-pikabu-2ch")
model = AutoModelForSequenceClassification.from_pretrained("../rubert-toxic-pikabu-2ch")

model_checkpoint = '../rubert-tiny-toxicity'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)


def text2toxicity(text, aggregate=True):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba




act = Softmax(dim=1)

def sentiment_analys(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    logits = model.forward(input_ids).logits
    labels = act(logits)
    #label = labels.argmax(axis=1).squeeze(0).item()
    return labels.squeeze(0)[1].item()

def return_vk_urls(key):
    vk_urls = []

    for title, snippet, url in get_search_engine(key, 0):
        if 'vk.com' in url:
            vk_urls.append(url)
    return vk_urls

def return_json(query):
    r = requests.get(query)
    if r.ok:
        json_content = json.loads(r.text)
        return json_content

def return_posts_vk(vk_urls):
    vk_url = vk_urls[0]

    if 'wall' in vk_url or 'public' in vk_url or 'club' in vk_url: 
        owner_id = re.search('[0-9]+', vk_url).group(0)
        query_wall = f'https://api.vk.com/method/wall.get?owner_id=-{owner_id}&count=20&access_token=4022960188fca3001ef01ef40225dccc46d44b32b92b49fee92&v=5.131'
        wall_json = return_json(query_wall)
        posts = []
        for post in wall_json['response']['items']:
            post_id = post['id']
            post_text = post['text']
            post_date = post['date']
            query_post_comment = f'https://api.vk.com/method/wall.getComments?owner_id=-{owner_id}&post_id={post_id}&access_token=4022960188fca3001ef01ef40225dccc46d44b32b92b49fee92&v=5.131'
            comments_json = return_json(query_post_comment)

            post_comments = []
            for comment_item in comments_json['response']['items']:
                post_comments.append(comment_item['text'])
            posts.append((post_text, post_date, post_comments))

    else:
        domain = vk_url.replace('https://vk.com/', '')
        query_wall = f'https://api.vk.com/method/wall.get?domain={domain}&count=20&access_token=4022960188fca3001ef01ef40225dccc46d44b32b92b49fee92&v=5.131'
        wall_json = return_json(query_wall)
        owner_id = wall_json['response']['items'][0]['owner_id']
        posts = []
        for post in wall_json['response']['items']:
            post_id = post['id']
            post_text = post['text']
            post_date = post['date']
            query_post_comment = f'https://api.vk.com/method/wall.getComments?owner_id={owner_id}&post_id={post_id}&access_token=4022960188fca3001ef01ef40225dccc46d44b32b92b49fee92&v=5.131'
            comments_json = return_json(query_post_comment)

            post_comments = []
            for comment_item in comments_json['response']['items']:
                post_comments.append(comment_item['text'])
            posts.append((post_text, post_date, post_comments))  
    return posts

def return_posts_df(key):
    vk_urls = return_vk_urls(key)
    
    if len(vk_urls) == 0:
        return ['Данная организация не найдена в сообществах ВКонтакте'],""
    else:
        posts = return_posts_vk(vk_urls)
        posts_df = pd.DataFrame(posts, columns=['text', 'date', 'comments'])
        posts_df['bank_cards'] = posts_df['text'].apply(lambda x: len(re.findall(r'[0-9]{4}\s*[0-9]{4}\s*[0-9]{4}\s*[0-9]{4}', x)))
        posts_df['sentiment'] = posts_df['comments'].apply(lambda comments: np.mean([sentiment_analys(comment) for comment in comments if comment != '']) if len(comments) != 0 else np.nan)
        return posts_df, vk_urls[0]
    
def return_mean_sentiment(df):
    mean_sentiment = df[~df['sentiment'].isnull()]['sentiment'].mean()
    return f'Средняя величина сентимента равна: {round(mean_sentiment, 2) * 100}%'

def return_bank_card_count(df):
    bank_cards_count = df['bank_cards'].sum()
    return f'Количество упоминаний банковских карт или реквизитов в последних 40 (если есть) постах на стене: {bank_cards_count}'

def return_last_post_days_from_now(df):
    days = (datetime.now() - datetime.utcfromtimestamp(df.loc[0, 'date'])).days
    return f'С момента публикации последнего поста прошло {days} дней'


def return_posts_agg_inform(key):
    posts_df, url = return_posts_df(key)

    if isinstance(posts_df, str):
        return posts_df
    
    elif isinstance(posts_df, pd.DataFrame):
        sentiment = return_mean_sentiment(posts_df)
        cards_count = return_bank_card_count(posts_df)
        last_post_days = return_last_post_days_from_now(posts_df)
        url = f'Ссылка на группу: {url}'
        
    return (url, sentiment, cards_count, last_post_days)

    
#################################################################################
def get_grant_features(inn):
    query = f'https://xn--80afcdbalict6afooklqi5o.xn--p1ai/public/application/cards?SearchString={inn}&Statuses%5B0%5D.Selected=true&Statuses%5B0%5D.Name=%D0%BF%D0%BE%D0%B1%D0%B5%D0%B4%D0%B8%D1%82%D0%B5%D0%BB%D1%8C+%D0%BA%D0%BE%D0%BD%D0%BA%D1%83%D1%80%D1%81%D0%B0&Statuses%5B1%5D.Name=%D0%BD%D0%B0+%D0%BD%D0%B5%D0%B7%D0%B0%D0%B2%D0%B8%D1%81%D0%B8%D0%BC%D0%BE%D0%B9+%D1%8D%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D1%82%D0%B8%D0%B7%D0%B5&Statuses%5B2%5D.Name=%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82+%D0%BD%D0%B5+%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B8%D0%BB+%D0%BF%D0%BE%D0%B4%D0%B4%D0%B5%D1%80%D0%B6%D0%BA%D1%83&RegionId=&ContestDirectionTenantId=&IsNormalTermProjects=true&IsLongTermProjects=true&CompetitionId=&DateFrom=&DateTo=&Statuses%5B0%5D.Selected=false&Statuses%5B1%5D.Selected=false&Statuses%5B2%5D.Selected=false&IsNormalTermProjects=false&IsLongTermProjects=false'
    main_request = requests.get(query, headers = {'User-Agent': 'Googlebot'})
    main_page = BeautifulSoup(main_request.text)
    
    all_features = {}

    for link in main_page.findAll('a'):
            features = {}
            if 'href' in link.attrs:
                if 'public/application/item?' in link['href']:
                    grant_link = 'https://xn--80afcdbalict6afooklqi5o.xn--p1ai' + link['href']
                    request = requests.get(grant_link, headers = {'User-Agent': 'Googlebot'})   
                    page = BeautifulSoup(request.text)

                    for li in page.findAll('li', class_='winner-info__list-item'):
                        feature_name = li.find('span', class_='winner-info__list-item-title').text
                        feature_value = li.find('span', class_='winner-info__list-item-text').text
                        feature_value = re.sub(r'\s+', ' ', feature_value).strip()
                        features[feature_name] = feature_value

                    for li in page.findAll('li', class_='circle-bar__info-item'):
                        feature_name = li.find('span', class_='circle-bar__info-item-title').text
                        feature_value = li.find('span', class_='circle-bar__info-item-number').text
                        feature_value = re.sub(r'\s+', ' ', feature_value).strip()
                        features[feature_name] = feature_value

                    text_description = page.findAll('div', class_='winner__details-box')

                    for block in text_description:    
                        feature_name = block.h2.text.strip()
                        feature_value = re.sub('\s+', ' ', block.text).replace(feature_name, '').strip()
                        features[feature_name] = feature_value
            if features:        
                all_features[grant_link] = features
                
    feat_df = pd.DataFrame.from_dict(all_features, orient='index')
    return feat_df

def normalize_decimal_cols(df):
    float_cols = ['Размер гранта', 
                  'Общая сумма расходов на реализацию проекта', 
                  'Cофинансирование', 
                  'Перечислено Фондом на реализацию проекта', 
                  'Рейтинг заявки']
    
    for col in float_cols:
        if col in list(df):
            if col == 'Рейтинг заявки':
                df[col] = df[col].apply(lambda x: float(x.replace(',', '.')) if x else x)
            else:
                df[col] = df[col].apply(lambda x: float(''.join(re.findall('[0-9]+', x.split(',')[0]))) if x else x)
    return df

def grant_info_flg(inn):
    grant_info = get_grant_features(inn)
    if len(grant_info) > 0:
        return True
    else:
        return False
    
    if isinstance(grant_info, pd.DataFrame):
        return True
    else:
        return False

#################################################################################


stop_words = get_stop_words('ru')


def remove_tags(lst):
    for i, text in enumerate(lst):
        tags = ['<passage>', '</passage>', '<hlword>', '</hlword>', '<title>', '</title>', '</url>', '<url>']
        for tag in tags:
            text = fix_text(text.replace(tag, ''))
        lst[i] = text
    return lst

def get_search_engine(key, page):
    query = f'http://xmlproxy.ru/search/xml?query={key}&groupby=attr%3Dd.mode%3Ddeep.groups-on-page%3D5.docs-in-group%3D3&maxpassages=1&page={page}&user=tumanov94%40gmail.com&key=bd65a9920e'
    r = requests.get(query)


    if r.status_code == 200:
        text = r.text
        titles = re.findall('<title>.*?</title>', text)
        snippets = re.findall('<passage>.*?</passage>', text)
        urls = re.findall('<url>.*?</url>', text)
        
    return list(zip(remove_tags(titles), remove_tags(snippets), remove_tags(urls)))

def return_pages_content(se_content):
    page_content = []

    for title, snippet, url in se_content:
        try:
            r = requests.get(url, headers={'User-agent': 'YandexBot'})
            if r.status_code == 200:
                content = BeautifulSoup(r.text)
                if content:
                    h1 = content.h1.text
                    h1 = re.sub('\s+', ' ', h1.strip())
                    page_text = re.sub('\s+', ' ', content.text).strip()
                    page_content.append((url, title, snippet, h1, page_text))
        except:
            pass

    content_df = pd.DataFrame(page_content, columns=['url', 'title', 'snippet', 'h1', 'page_text'])
    return content_df

def preprocess_text(text):
    tokenizer = RussianTokenizer()
    tokens = tokenizer([text])
    morph = pymorphy2.MorphAnalyzer()
    tokens = [morph.parse(token)[0].normal_form for token in tokens[0] if token not in stop_words and token != " " \
              and token.strip() not in punctuation]
    return tokens

def find_fraud_phrases(df):
    fraud_phrases = pd.read_csv('fraud_phrases.txt')
    tokenizer = RussianTokenizer()
    match_phrases = []
    all_content = ''
    for row in df.values:
        text_content = ' '.join(row[1:])
        for fraud_phrase in fraud_phrases['phrase'].unique():
            match_phrases.extend(re.findall(f' {fraud_phrase}', ' '.join(preprocess_text(text_content))))
            all_content += ' '.join(preprocess_text(text_content))
    match_phrases = [phrase.strip() for phrase in match_phrases]
    return match_phrases, len(tokenizer([all_content])[0])


def find_fraud_in_SE(key):
    se_content = get_search_engine(key, 0)
    df = return_pages_content(se_content)
    fraud_phrases = find_fraud_phrases(df) 
    return fraud_phrases

def fraud_or_not(key):
    fraud_triggers, token_count = find_fraud_in_SE(key)
    find_tags = ', '.join([trigger for trigger in set(fraud_triggers)])

    if token_count > 0:
        fraud_density = (len(fraud_triggers) / token_count) * 100
    else:
        print("token_count = 0!")
        fraud_density = 0

    if fraud_density > 0.01:
        return f'Это мошенническая организация! Найдены следующие теги в поисковой выдаче: {find_tags}.'
    else:
        return 'Результат анализа поисковой выдачи показал, что данная организация не мошенническая.'

#############################################################################

def dobrma(org_name):
    org_name = org_name.replace(' ', '+')
    urls = f'https://dobro.mail.ru/funds/search/?query={org_name}&recipient=all&city=any'
    response = requests.get(urls)
    soup = BeautifulSoup(response.text)
    req = soup.find_all('div', class_='cols__column cols__column_small_percent-100 cols__column_medium_percent-50 cols__column_large_percent-50')
    if len(req) != 0:
        for i in range(len(req)):
            name = str(req[i].find('span', 'link__text'))
            name = name.replace('<span class="link__text">', '')
            name = name.replace('</span>', '')
            url = str(req[i].find('a', 'link link_font_large margin_bottom_10 link-holder'))
            url = url.replace('<a class="link link_font_large margin_bottom_10 link-holder" href="', 'https://dobro.mail.ru')
            url = url.split('"')[0]
            city = str(req[i].find('div', 'p-fund__city margin_bottom_5'))
            city = city.replace('<div class="p-fund__city margin_bottom_5">', '')
            city = city.replace('</div>', '')
            responses = requests.get(url)
            soups = BeautifulSoup(responses.text)
            site = soups.find_all('div', class_='p-fund-detail__info-row')
            if str(site[0].find('a', 'link')) == None: site = str(site[0].find('a', 'link')); site = site.replace('<span class="link__text">', ''); site = site.replace('</span>', '')
            else: site = str(site[1].find('a', 'link')); site = site.replace('<span class="link__text">', ''); site = site.split('"')[3]
        return site
    else: 
        return False

def isin_dobrmail(org_name):
    if dobrma(org_name):
        return f'Организация {org_name} присутствует на сайте https://dobro.mail.ru/'
    else:
        return f'Организация {org_name} отсутствует на сайте https://dobro.mail.ru/'

def return_vse_meste_orgs():
    urls2 = 'https://wse-wmeste.ru/about/uchastniki-assotsiatsii/'

    vse_vmest = []
    desc = []
    response2 = requests.get(urls2)
    soup3 = BeautifulSoup(response2.text)
    query3 = soup3.find('div', 'col-md-12')
    query3 = query3.find_all('div', 'wp-block-themeisle-blocks-posts-grid-post-blog wp-block-themeisle-blocks-posts-grid-post-plain')

    for i in range(len(query3)):
        name = str(query3[i].find('h5', 'wp-block-themeisle-blocks-posts-grid-post-title text-center')).split('>')[2].replace('</a', '')
        disc = str(query3[i].find('p', 'wp-block-themeisle-blocks-posts-grid-post-description')).replace('<p class="wp-block-themeisle-blocks-posts-grid-post-description">', '').replace('</p>', '')
        vse_vmest.append(' '.join(re.findall('\w+', fix_text(name.lower()).replace('\xa0', ' ')))) 
        desc.append(fix_text(disc).replace('\xa0', ' '))
    return vse_vmest
    
def find_in_vse_vmeste(name_org, all_orgs):
    name_org = ' '.join(re.findall(r'\w+', name_org.lower()))
    if name_org in all_orgs:
        return f'Организация "{name_org.capitalize()}" есть на сайте https://wse-wmeste.ru/about/uchastniki-assotsiatsii/'
    else: 
        return f'Организация "{name_org.capitalize()}" отсутствует на сайте https://wse-wmeste.ru/about/uchastniki-assotsiatsii/'
    

def find_nuzna_pomosh(org_name):
    org_name = ' '.join(re.findall('\w+', org_name))
    url = 'https://nuzhnapomosh.ru/wp-content/plugins/nuzhnapomosh/funds.php'
    typ = 'POST'
    req = Request(typ, url, files = {'np_name': (None, org_name)}).prepare()
    s = Session()
    respo = s.send(req);respo.text
    soup2 = BeautifulSoup(respo.text)
    quotes2 = soup2.find_all('div', class_='np-card__inner')

    if len(quotes2) != 0:
        for i in range(len(quotes2)):
            name = str(quotes2[i].find('h4', 'np-card__title'));  name = name.replace('<h4 class="np-card__title">', '');  name = name.replace('</h4>', '')
            descr = str(quotes2[i].find('p', 'np-card__descr'));  descr = descr.replace('<p class="np-card__descr">', '');  descr = descr.replace('</p>', '')
            money = str(quotes2[i].find('li', 'np-card__row'));  money = money.split('<span>')[1];  money = money.replace('</li>', '');  money = money.replace(' ₽</span>', '');money = money.replace('\n', '')
            site = str(quotes2[i].find('a', 'np-card__link'));  site = site.replace('<a class="np-card__link" href="', '');  site = site.split('"')[0]
        return (f'Организация "{name}" найдена на сайте https://nuzhnapomosh.ru{site}',
                'Описание фонда: ' + descr, 
                'Деньги в фонде: ' + money + ' рублей')
    else: 
        return f'{org_name} не найдена https://nuzhnapomosh.ru'    

def find_in_reestr_sonko(inn):
    reestr_sonko = pd.read_excel('reestr_SONKO.xlsx')
    if inn in reestr_sonko['ИНН'].unique():
        status = reestr_sonko.loc[reestr_sonko['ИНН'] == inn, 'Текущий статус'].values[0]
        return f'Найдено в реестре социально значимых НКО, текущий статус: {status.lower()}'
    else:
        return 'Не найдено в реестре социально значимых НКО'
    
    
#############################################################################



app = Flask(__name__, static_url_path='', 
            static_folder='/home/dsvm/perm/flask/templates',
            template_folder='/home/dsvm/perm/flask/templates')


def check_extrim_terr(inn):
    r = requests.post("https://www.fedsfm.ru/TerroristSearch", data={"rowIndex":0,"pageLength":50,"searchText":inn})
    
    # print(r.status_code, r.reason)
        
    if r.status_code == 200:
        resp_json = json.loads(r.text)
        
        #print(resp_json['recordsTotal'])
        #print(resp_json)
            
        if resp_json['recordsTotal'] == 1:
            return(' тип реестра: ' + resp_json['data'][0]['TerroristTypeName'])
    else:
        return None

def get_dadata(inn):
    # security fail =]
    token = "fcdc5c2aff4b6e419bde299da262278181b39bf6"
    dadata = Dadata(token)
    
    return dadata.find_by_id("party", inn)

# 7709471429 - fbk
# 7708236775 - habenskiy

## global for get reports links
reports = None

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    
    if request.method == 'POST':
#         f = request.files['input_file']
#         f.save('./ustav_v/curr.pdf')
        pass
    
    inn_info = ""
    inn_terr = ""
    
    ### 2DO dadata умеет работать и ИНН, и с ОГРН, можем конвертировать
    
    # inn = request.args.post('inn')
    inn = request.form.get('inn')
    
    inn_terr = None
        
    inn_nalog = None
    
    inn_info = None
    
    df = None
    
    percents = None
    
    pls = None
    
    fraud = None
    
    vk = None
    
    grants = None
        
    if inn is not None:
        print("inn:")
        print(inn)
        
        inn_terr = check_extrim_terr(inn)
        
        inn_nalog = get_dadata(inn)
        
        if len(inn_nalog) == 0:
            pass
        
        print(type(inn_nalog))
        
        inn_nalog = inn_nalog[0]
        
        print(inn_nalog)
        print(type(inn_nalog))
        
        
        import datetime

        # print(inn_nalog)      
        
        # print(inn_nalog['name'])
        
#         inn_info = ""
        
#         inn_info+= "<b>полное название </b>" + inn_nalog['data']['name']['full_with_opf'] + "<br>"
        
#         inn_info+= "<b>сокращённое название </b>" + inn_nalog['data']['name']['short_with_opf'] + "<br>"
    
        inn_nalog['data']['ogrn_date'] = datetime.datetime.fromtimestamp(inn_nalog['data']['ogrn_date'] / 1e3).date()
        inn_nalog['data']['state']['registration_date'] = datetime.datetime.fromtimestamp(inn_nalog['data']['state']['registration_date'] / 1e3).date()
        inn_nalog['data']['state']['actuality_date'] = datetime.datetime.fromtimestamp(inn_nalog['data']['state']['actuality_date'] / 1e3).date()
                
        #################################
        #################################
        
        pls = []
        
        pls.append(isin_dobrmail(inn_nalog['data']['name']['short_with_opf']))
        

        vse_vmeste_orgs = return_vse_meste_orgs()
        
        pls.append(find_in_vse_vmeste(inn_nalog['data']['name']['short_with_opf'], vse_vmeste_orgs))
        
        pls.append(find_nuzna_pomosh(inn_nalog['data']['name']['short_with_opf']))
        
        pls.append(find_in_reestr_sonko(int(inn_nalog['data']['inn'])))
        
        fraud = fraud_or_not(inn_nalog['data']['name']['short_with_opf'])
        
        print(fraud)
        
        print(inn_nalog['data']['name']['short_with_opf'])

        vk = return_posts_agg_inform(inn_nalog['data']['name']['short_with_opf'])

        
        #####################################################
        
        if grant_info_flg(int(inn_nalog['data']['inn'])):
            print("Гранты найдены!")
            df = get_grant_features(inn)
            df = normalize_decimal_cols(df)
            df.reset_index(inplace=True)
            print(df.values)
            
            grants = list(df[['index', 'Рейтинг заявки', 'Размер гранта', 'Cофинансирование', 'Цель', 'Сроки реализации']].values)

        ####################################################################################
        ### minust 
        ##
        #
        
        ogrn = inn_nalog['data']['ogrn']
        year = 2021
    
        def get_minust_reports(ogrn, year):
            import subprocess
            
            try:
                py2output = subprocess.check_output(['/home/dsvm/perm/go/make_parse', '-psrn', str(ogrn), '-y', str(year)])
            except BaseException:
                return None
            
            final_pdfs = []
            
            for u in py2output.split():
                final_pdfs.append(u.decode("utf-8").replace('\n',''))
            
            final_pdfs = list(set(final_pdfs))
        
            return final_pdfs
        
        def save_minj_report(report_url):
            file_name = report_url.split('/')[-1]
            r = requests.get(report_url, headers = {'User-Agent': 'YandexBot'})
            file_path = f'../minj_reports/{file_name}'
            
            with open(file_path, 'wb') as f:
                f.write(r.content)
                
            return file_path
        
        def read_pdf(path):
            data = parser.from_file(path)
            data = re.sub(r'\n+', '\n', data['content']).split('\n')
            return data
        

        
        def find_need_report(ogrn, year):
            global reports
            reports = get_minust_reports(ogrn, year)
            
            print(reports)
            
            if not reports:
                return 'На сайте Минюста не найдены отчеты за 2021 год'
            
            else:
                import glob, os
                test = f'../minj_reports/*.pdf'
                r = glob.glob(test)
                print("remove")
                for i in r:
                    print(i)
                    os.remove(i)

                for report_url in reports:
                    save_minj_report(report_url)
        
                reports_folder = f'../minj_reports/'
        
                files = os.listdir(reports_folder)
        
                for file_path in files:
        
                    data = read_pdf(os.path.join(reports_folder, file_path))
                    data = [line for line in data if line != '']
                    for row in data:
                        if 'Форма: O H 0 0 0 2' in row:
                            break
                            
                return data
        
        def return_decimal_lines(data_lst):
            dec_lines = []
            
            for row in data_lst:
                if row:
                    # match_str = re.match(r'[0-9.\s]+\s+([А-я\s.,]+) ([0-9]+)', row)
                    match_str = re.match(r'[0-9\.\s]+\s+([А-Яа-я\s\.,"]+) ([0-9]+)', row)
                    if match_str:
                        name = match_str.group(1)
                        decimal = match_str.group(2)
                        dec_lines.append((name, decimal))
            return dec_lines
        
        def return_report_features(ogrn, year):
            data = find_need_report(ogrn, year)
            if isinstance(data, list):
                decimal_lines = return_decimal_lines(data)
                if decimal_lines:
                    df = pd.DataFrame(decimal_lines, columns=['feature', 'value'], dtype='float64')
                    return df.groupby('feature')['value'].sum().reset_index().sort_values(by='value', ascending=False)
                else:
                    return 'В отчете нет строк удовлетворяющих условию!'
            else:
                return 'Отчетов на сайте Минюста, удовлетворяющих условию нет!'
            
        def return_percents(df):
            if isinstance(df, str):
                return df
            else:
                percents = df.loc[df['feature'] == 'Социальная и благотворительная помощь', 'value'] / df['value'].sum()
                return 'Расходы на социальную и благотворительную помощь организации составляет ' + str(round(percents.values[0] * 100, 2)) + '% от всех расходов!'
            
        df = return_report_features(ogrn, year)
        
        #percents = return_percents(df)
        
        if isinstance(df, str):
            pass
            tmp_df = df
            df = None
        else:
            df = dict(df.values)
        
    
    #### company number
    c = request.args.get('c')
    
    if c is not None:
        print("!")
        print(c)
    else:
        c = 0

    
    company = companies[int(c)]
        

    return render_template('main.html', posts=companies, company=company, inn_info=inn_nalog, inn_terr = inn_terr, inn = inn, percents = percents, df = df, reports=reports, pls = pls, fraud=fraud, vk=vk, grants=grants)

if __name__ == '__main__':
    app.run()