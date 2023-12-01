import requests
import pandas as pd
from pandasai.llm import OpenAI
from pandasai import PandasAI


def main():
    print("Here is your data:")
    print(get_data_frame())
    while True:  # Continually prompt unless user Quits
        print("what would you like to do?(Enter an integer)")
        print("1. Download data as csv\n2. Ask AI questions about data\n3.Quit")
        user_input = input()
        if user_input == "1":
            to_csv()
        elif user_input == "2":
            AI_driver()
        elif user_input == "3":
            print("Exiting...")
            exit()  # Exit the loop when the user selects option 3
        else:
            print("Try again.")

#Allows user to interact with AI model directly through command line
def AI_driver():
    print("Enter your API key to access AI ")
    your_api_key = input("")
    llm = OpenAI(api_token=your_api_key)
    pandas_ai = PandasAI(llm)

    print("Ask AI about your data. Type \"Q\" to stop.")

    while True:
        user_query = input()
        if user_query.lower() == "q":
            return  # Exit the loop if the user types "Q" or "q"
        else:
            print(pandas_ai(get_data_frame(), prompt=user_query))

#converts a data frame to a csv and downloads it
def to_csv():
    data_frame = pd.json_normalize(get_data_frame())
    data_frame.to_csv('sunglasses.csv')
    print("sunglasses.csv was saved into your files\n")
    return

#Gets data from sunglass hut web page
#returns a data frame
def get_data_frame():
    # page to scrape
    url = "https://www.sunglasshut.com/wcs/resources/plp/10152/byCategoryId/3074457345626651837"

    result = []
    # loop through page 1-5... change to however many pages you want
    for x in range(1, 6):
        # generated in insomnia by copying the cURL of the json data package
        querystring = {"isProductNeeded": "true", "isChanelCategory": "false", "pageSize": "100", "responseFormat": "json", "currency": "USD", "viewTaskName": "CategoryDisplayView", "storeId": "10152",
                       "DM_PersistentCookieCreated": "true", "pageView": "image", "catalogId": "20602", "top": "Y", "beginIndex": "0", "langId": "-1", "categoryId": "3074457345626651837", "orderBy": "default", "currentPage": f"{x}"}

        payload = ""
        headers = {
            "cookie": "aka-cc=US; aka-ct=PHOENIX; aka-zp=85001-85046+85048+85050-85051+85053-85055+85060-85076+85078-85080+85082+85083+85085-85086+85097-85098; bm_sz=FFCE5E1EB28B438EC2D064B0B72A26F4~YAAQH608FyIkUvaJAQAA5q16+hT2IW68l/tLhI9qpVrDAPbUDw8UMy5qO8irm4EsjFsjXlOmhj2hUgV+4je/jUIvlA/NNuZHmGin4EwwxwMEXmYHAcy5IdSZWrRegZNRAyZbo1fEhUOI+I++8oATB0iubLYVi+PyBo06lEMvWgZkLqY7cir+dzKV+nvGudQEWff09bsEoaVOeAdyvj9Or6rmUGfEmYuYWvZ0Bp4+QY+V0Jw8/j3koEot2F3610mHsMAIkJsZO2bCwqyjfmnU4YrgpkUfKWtT03yN1WJwzk4LPL0e0164sg==~4276802~3617336; dtCookie=v_4_srv_-2D11_sn_US42NI4J48P809O98QGOL12BCOSBN5C7; rxVisitor=1692124491629ESUCDSCB6N13B1C33C8J51VUQA40SGDR; mt.v=2.211937119.1692124491709; sgh-desktop-facet-state-search=; JSESSIONID=00000VYul9d7udcUUxqXpHPMrbI:1c7qtpr06; tealium_data_tags_adobeAnalytics_trafficSourceCid_lastFirst=PM-SGA_300419-1.US-SGH-EN-B-BrandCore-Exact_Pure-E_sunglass%2520hut; tealium_data_tags_adobeAnalytics_trafficSourceCid_thisSession=PM-SGA_300419-1.US-SGH-EN-B-BrandCore-Exact_Pure-E_sunglass%2520hut; tealium_data2track_Tags_AdobeAnalytics_TrafficSourceMid_ThisHit=PM-SGA_300419-1.US-SGH-EN-B-BrandCore-Exact_Pure-E_sunglass%2520hut; tealium_data_tags_adobeAnalytics_trafficSourceMid_thisSession=PM-SGA_300419-1.US-SGH-EN-B-BrandCore-Exact_Pure-E_sunglass%2520hut; tealium_data_session_timeStamp=1692124492165; userToken=undefined; tiktok_click_id=undefined; ftr_ncd=6; ak_bmsc=70F93F5EBCA71E4DEB7BE4AB244535D7~000000000000000000000000000000~YAAQH608F1UkUvaJAQAA4LJ6+hRTHUmHL/xcLkVTkVe8mZ7NK7aHN4IyNABkkUM+weJsuUQh8lfui+o7Ky5KvvQcxBfvM7WdighTU+i5iLBi+B9YycVRouOkoWPNSPfzXYvMGISwdi9rKuYrqHbDbtvfro4M8krOqQtzkCARqgLc+7gjuTzP7aUX7F2Tty5na3Loz2h/9jI6/2KCPVu/QF5lMadXuRtH9UUHZH1ALPq+535AMO7BFGS/LPopjZdIgYJSK+q5lAXnop7nNPs0MSryVlXA53/4o70y+7wfJHOpSdpGwGg65hJbaaxQ92pp4sWnBdROBqllSaYXqewoW1oue/iHOLK8CbRPQ5EAZKek+5tnXZWixBSVy7p/WO2zNJ1e/9ztD1OBNlkGv7xicKTT7jGyjRnN0fKM1tGk5x7EXdHywP9YNHdECs++1pkzUpof+u545PcdWX1rhRZD0N9nKgtBFHPZU5CyL9QIBr57DA8dx2dE4wndj2xCjClCeI4VF1R8OVng2Tdkx2JAiTXsKUCpp43M5g==; __wid=681322343; ftr_blst_1h=1692124492492; AMCVS_125138B3527845350A490D4C%40AdobeOrg=1; s_ecid=MCMID%7C03860413034861960901632383693018687637; AMCV_125138B3527845350A490D4C%40AdobeOrg=-1303530583%7CMCIDTS%7C19585%7CMCMID%7C03860413034861960901632383693018687637%7CMCAAMLH-1692729292%7C9%7CMCAAMB-1692729292%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1692131692s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C3.3.0; s_cc=true; CONSENTMGR=consent:true%7Cts:1692124493239; _cs_mk_aa=0.9316400028825089_1692124493261; _cs_c=1; UserTrackingProducts=0; _gcl_au=1.1.2064494104.1692124495; _ga=GA1.1.1325316750.1692124495; _pin_unauth=dWlkPU16QTBPVE13T0dJdFltSTVNUzAwWmpFd0xUZzJaVGt0WmpFNVpURTJNemt6TUROaw; _scid=1659b773-ede4-40c2-bba7-c3a01225070a; __pdst=96a4e08c56b74b58bef1c7d0f45bea12; _abck=484BA9BA0ADB6D7679EEECCE66073282~0~YAAQH608F98kUvaJAQAA5L56+goBBOGnfUtmCPBbMgPIGsFM8VBxm2snYDCb/iz6vQqyxMb8H7agJra/vNc3fcP3GAmQzfsJwjRC5qO/fUakfl+D6K1Fmjhsrh6UCuuOyKCtkggNP6gDJNmFrjeYpmwdUygLAxWIO1jr0YUOb9vXBVrD2pnWTsTg3FsuWRpDT0c7ySohsPT/mLbt9MVzXVz6cmvEI3gNbmQ/Pvau7NEpKw9DSRWRb/tMwJZ5cwL3sy8Ha1hR5xz6YdT5peGnR796K9NHEcOyqwa9hd08j2MbNG4q4J1gZKjy7/ve+9+KDyS1IqKsZv9eODhZ5VUqyIX8dMPKheZVOTXOgrsllhFlL2/RdujZILtqUSpcST9sqzBArIgZit475gpNdDsH14LCaAJjRODi3S9qX2Q=~-1~-1~-1; __rtbh.lid=%7B%22eventType%22%3A%22lid%22%2C%22id%22%3A%22ia7vj9NmUBx42doJFV9V%22%7D; _fbp=fb.1.1692124495509.1851430010; _tt_enable_cookie=1; _ttp=1NI9rO4WFt1qmnFbd2x-ZEYiLHI; _screload=1; tfpsi=7f6add94-db9e-4740-ab4b-2f0cebd00fa1; __idcontext=eyJjb29raWVJRCI6IjJVMjRsdmxLOTNYUEVCZW5XVGFSbGkwZjI2RSIsImRldmljZUlEIjoiMlUyNGx2YkxneW9VSDcwR2RZckgzNEV5T3FaIiwiaXYiOiIiLCJ2IjoiIn0%3D; smc_uid=1692124496074867; smc_tag=eyJpZCI6Njc4LCJuYW1lIjoic3VuZ2xhc3NodXQuY29tIn0=; smc_session_id=QpO5FNPYIqkCNK20obn5K8J6yDjRbtoi; _sctr=1%7C1692082800000; smc_refresh=24860; smc_sesn=1; smc_not=default; SGPF=3XEaLav6u5j0s4COG4QsrDm8jMBFvAeHIDiy7QG5hHDGi14o34xy56w; tealium_data_action_lastEvent=click [sgh-load-more  button][Loadmoresunglasses]; s_sq=%5B%5BB%5D%5D; TS011f624f=015966d29221104ca48fed7f8ab3a2c9bf9c80d8aaa16331303ebe9d9a5c28cd6b0d7d0844d39fb48510b56812e3c8249eba362195106c1dad4936772327e26a29ce79a3bb; sgh-desktop-facet-state-plp=categoryid:undefined|gender:true|brands:partial|polarized:true|price:true|frame-shape:partial|color:true|face-shape:false|fit:false|materials:false|lens-treatment:false; dtLatC=12; dtSa=-; tealium_data2track_Tags_AdobeAnalytics_TrafficSourceJid_ThisHit=308154REF; tealium_data_tags_adobeAnalytics_trafficSourceJid_stackingInSession=308151SEM-308152REF-308153REF-308154REF; bm_sv=9F40F49691778101B35FF66B39676A8F~YAAQH608F+7JUvaJAQAAFnyR+hT43sACLZGkTnCr18NAuMuSDfB4y6020JNWekDwaM9KJYshy07RrusUP3SJfJlLcu4U3Ez2OqOL0aYJzNdbHKIudInuA3Agq64uR6i5wJpC9eSBnqUClGxsmAhFaZEhRB7dRXqxfGIKj6JDWkvaQiHImqcU7druuYjV3lEsrFKgGwe7XJ7kgGP3zgt+qHfnnWJkSnqbNeRRVST13KEd3c+hQIguTjM+CCKlDM+9vPhQU/GL~1; _cs_id=226f6799-0e41-a4d6-e625-f2bfb5390898.1692124493.1.1692125986.1692124493.1.1726288493436; _cs_s=4.0.0.1692127786039; _uetsid=6e13f4703b9a11ee9a4e01bf375c72f9; _uetvid=6e1415b03b9a11eea996cf1da6242ece; _scid_r=1659b773-ede4-40c2-bba7-c3a01225070a; outbrain_cid_fetch=true; cto_bundle=1tRsb18zb2o4WGRoN0E1cnBMMVRnV3VVQTk3ekxvWjNyJTJGYnlRYmFJT3NSNUNmZFhBTlQ1ZiUyQnBCWHRxbWhybnRYQjRRNXB3bUs5VEhJS2ZhMkNWeE12NUJEZWp6MSUyRjc1d2NuR3oxTkVjNDdrdDVJdjAxTjhKaW9ibDk4JTJGJTJGTzRoQVVJT2J3MzNXJTJCaWlnV3FYODBQbiUyRkxCaFJDR0pocXd3VWglMkJlSDl4d1RHJTJGOUpqb1Uwd3pvNnN6cDNUMVZlMVRONkwza2dXYWc0QWklMkJuWmJkWnVUUU9KbmRDREElM0QlM0Q; forterToken=bea45feecf3644f6817d339b7283a614_1692125984681__UDF43-m4_6; smc_tpv=5; smc_spv=5; rxvt=1692127788406|1692124491630; dtPC=-11$125984561_25h-vCHHKRVFBKQFDUUQKIAKRLQSSOOBDHDLH-0e0; _ga_6P80B86QTY=GS1.1.1692124495.1.1.1692126003.40.0.0; utag_main=v_id:0189fa7ab17700639da9ce54a47005075013306d00942$_sn:1$_se:16$_ss:0$_st:1692127805811$ses_id:1692124492152%3Bexp-session$_pn:4%3Bexp-session$vapi_domain:sunglasshut.com$dc_visit:1$dc_event:4%3Bexp-session$dc_region:us-west-2%3Bexp-session; tealium_data_action_lastAction=Men:Sun:Plp click [sgh-load-more  button][Loadmoresunglasses]; smct_session={\"s\":1692124497082,\"l\":1692126005925,\"lt\":1692126005925,\"t\":1410,\"p\":712}",
            "authority": "www.sunglasshut.com",
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "referer": "https://www.sunglasshut.com/us/mens-sunglasses?currentPage=2",
            "sec-ch-ua": "\"Not/A)Brand\";v=\"99\", \"Google Chrome\";v=\"115\", \"Chromium\";v=\"115\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

        response = requests.request(
            "GET", url, data=payload, headers=headers, params=querystring)
        data = response.json()

        # For each product on the given page, apppend its json data to result
        for i in data['plpView']['products']['products']['product']:
            result.append(i)

    print(f"number of products: {len(result)}")
    return pd.DataFrame(result)


if __name__ == "__main__":
    main()
