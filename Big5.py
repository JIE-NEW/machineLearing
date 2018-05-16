# -*- coding: utf-8 -*-
__author__ = 'Jiao'
import sys
import ssl
import urllib2

ssl._create_default_https_context = ssl._create_unverified_context

reload(sys)
sys.setdefaultencoding('utf-8')

url = 'https://ccpl.psych.ac.cn:20027/big5?AppKey=PXC387343a67cU7N1FE' #需要申请AppKey
data = """[
            {"id":123,
             "gender":1,
              "age":53,
              "answers":["我我我我我我我我哦我我我我我我我我我我我我我我我我我我我",
              "我，我，我，我，我，我，我，我，哦，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我。",
              "我，我，我，我，我，我，我，我，哦，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我，我。"]
            },
            {"id":124,
             "gender":1,
              "age":53,
              "answers":["我们都不了解的隐秘江湖，三十六计，为上计。 //@信天游唱不出的凄凉:财新的这篇基本靠谱",
              "如果在自己家摔倒，找谁赌？翡翠是办理银行业的必然品吗？"]
            },
            {"id":125,
             "gender":1,
              "age":53,
              "answers":["我们，我们，我们，我们，我们，我们，我们，我们，我们，我们，我们，我们，我们",
              "我们，我们，我们，我们，我们，我们，我们，我们，我们，我们，我们，我们，我们道长.集体 ，奉献，牲，，关爱.. "]
            }
        ]"""
print type(data)

req = urllib2.Request(url, data)
response = urllib2.urlopen(req)
the_page = response.read()
print the_page 
print data
