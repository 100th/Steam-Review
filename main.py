import urllib
import urllib.request
import urllib.parse
import bs4
import re
import os
from concurrent.futures import ThreadPoolExecutor


def deleteTag(x):
    return re.sub("<[^>]*>", "", x)


def getComments(appid, lang='all'):
    def makeArgs(appid, lang, page):
        params = {
            'userreviewsoffset': 10 * (page - 1),
            'p': page,
            'workshopitemspage': page,
            'readytouseitemspage': page,
            'mtxitemspage': page,
            'itemspage': page,
            'screenshotspage': page,
            'videospage': page,
            'artpage': page,
            'allguidepage': page,
            'webguidepage': page,
            'integratedguidepage': page,
            'discussionspage': page,
            'numperpage': 10,
            'browsefilter': 'toprated',
            'appid': appid,
            'appHubSubSection': 10,
            'l': 'english',
            'filterLanguage': lang,
            'searchText': '',
            'forceanon': 1
        }
        return urllib.parse.urlencode(params)

    def innerHTML(s, sl=0):
        ret = ''
        for i in s.contents[sl:]:
            if i is str:
                ret += i.strip()
            else:
                ret += str(i)
        return ret

    def fText(s):
        if len(s): return innerHTML(s[0]).strip()
        return ''

    retList = []
    colSet = set()

    print("Processing: %d" % appid)
    page = 1
    while 1:
        try:
            f = urllib.request.urlopen(
                "http://steamcommunity.com/app/" + str(appid) + "/homecontent/?" + makeArgs(appid, lang, page))
            data = f.read().decode('utf-8')
        except:
            break
        soup = bs4.BeautifulSoup(data, "html.parser")
        cs = soup.select(".apphub_Card")
        if not len(cs): break
        for link in cs:
            url = link.get('data-modal-content-url')
            url = url[29:].replace('recommended/', '')
            if url in colSet: return retList
            colSet.add(url)
            helpful = fText(link.select('.found_helpful'))
            cat = fText(link.select('.title'))
            cont = innerHTML(link.select('.apphub_CardTextContent')[0], 2)
            cont = deleteTag(re.sub("([\t\r\n ]|<br>|</br>)+", ' ', cont)).strip()

            ng = 0
            nb = 0
            nf = 0
            nh = re.search("^([0-9]+) of ([0-9]+)", helpful)
            if nh:
                ng = int(nh.group(1))
                nb = int(nh.group(2)) - ng
            nh = re.search("([0-9]+) people found this review funny", helpful)
            if nh:
                nf = int(nh.group(1))
            retList.append((url, cat, cont, ng, nb, nf))
        page += 1

    return retList


def fetch(i):
    outname = 'comments/%d.txt' % (i * 10)
    try:
        if os.stat(outname).st_size > 0: return
    except:
        None
# english 대신에 korean을 넣으면 한국어를 긁어옵니다.
    rs = getComments(i * 10, 'english')
    if not len(rs): return
    f = open(outname, 'w', encoding='utf-8')
    for r in rs:
        f.write('%d\t%s\t%s\t%s\t%d\t%d\t%d\n' % (i * 10, r[0], r[1], r[2], r[3], r[4], r[5]))
    f.close()


with ThreadPoolExecutor(max_workers=5) as executor:
#일단 게임ID가 어디서부터 어디까지인지 몰라 다음과 같이 지정했습니다.
    for i in range(1, 100): #100000
        executor.submit(fetch, i)
