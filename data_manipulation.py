import pandas as pd

# pandas kutuphanesi ile bilgisayardaki csv dosyasindan veri seti okuma
# dataset link :   https://archive.ics.uci.edu/dataset/162/forest+fires
veriSeti = pd.read_csv("forestfires.csv", sep=",")

#veriseti özelliklerini inceleme
veriSeti.info()
veriSeti.dtypes
print(veriSeti.shape)

# Kategorik degiskenlerin veri tipinin category yapilmasi
veriSeti["ay"] = veriSeti["ay"].astype("category")
veriSeti["gun"] = veriSeti["gun"].astype("category")

veriSeti.head(5)
veriSeti.tail(5)
  
#veriseti değişken isimlerini değiştirme
veriSeti.rename(
    columns={"X": "X_kordinatı", "Y": "Y_koordinatı", 
             "month": "ay", "day": "gun", "temp": "sıcaklık",
             "RH": "bagıl_nem", "wind": "ruzgar_hızı",
             "rain": "yağmur_miktarı", "area": "yanan_alan"},
    inplace=True,
)  

veriSeti["ay"].value_counts()
veriSeti["ay"] = veriSeti["ay"].replace(["aug", "sep", "mar", "jul", "feb",
                                         "jun", "oct", "apr", "dec", "sept",
                                         "jan", "may", "nov"],
                                        ["ağustos", "eylül", "mart",
                                         "temmuz", "şubat", "haziran",
                                         "ekim", "nisan", "aralık", "eylül",
                                         "ocak", "mayıs", "kasım"])
veriSeti["ay"].value_counts() 


veriSeti["gun"].value_counts()
veriSeti["gun"] = veriSeti["gun"].replace(["sun", "sat", "fri", "mon", "tue",
                                           "thu", "wed", "tuesday"],
                                          ["pazar", "cumartesi", "cuma",
                                          "pazartesi", "salı", "perşembe",
                                           "çarşamba", "salı"])
veriSeti["gun"].value_counts()



veriSeti.describe()  # sadece sayisal degiskenlerin istaitstikleri

pd.set_option('display.max_columns', None) #consol üzerinde tüm veriyi görmek için yazılır

veriSeti.describe(include="all")


# null değer olan kısımları doldurma
veriSeti.isnull().any()
veriSeti.isnull().sum()

median_DMC = veriSeti['DMC'].median()
veriSeti['DMC'].fillna(median_DMC, inplace=True)

median_bagil_nem = veriSeti['bagıl_nem'].median()
veriSeti['bagıl_nem'].fillna(median_bagil_nem, inplace=True) 

# mode_ay = veriSeti['ay'].mode() --> kategorik veri için en sık görülen kategoriyi bulma

veriSeti.isnull().any()

# feature selection
veriSeti=veriSeti.drop(["X_kordinatı"], axis = 1)
veriSeti=veriSeti.drop(["Y_koordinatı"], axis = 1)

#veriSeti = veriSeti.drop(columns=['X_ekseni', 'Y_ekseni'])


# ALT KUMELERE AYIRMA (Subsetting)
# ağustos ayındaki yangınlar
subset1_1 = veriSeti.loc[veriSeti.ay == "ağustos",] 
subset1_2 = veriSeti.loc[veriSeti.sıcaklık > 19.3,]

# yanan alan 1 hektar üstünde olan ve pazar günü kaydedilen yangınlar
subset2 = veriSeti[(veriSeti.yanan_alan > 1) & (veriSeti.gun == "pazar")] # Birinci yol

# GRUPLANDIRMA (Aggregate)

veriSeti.loc[veriSeti.ay == "ağustos", "sıcaklık"].mean()
veriSeti.loc[veriSeti.gun == "cumartesi", "DMC"].mean()


veriSeti.groupby(["ay"])["ISI"].aggregate("mean")

veriSeti.loc[veriSeti["gun"]=="pazar", "yanan_alan"].describe()

# Gruplar arasi karsilastirma icin
veriSeti[["ruzgar_hızı", "ay"]].groupby("ay").describe()
veriSeti[["yağmur_miktarı", "gun"]].groupby("gun").describe()



# VERI AYRIKLASTIRMA
#1
ruzgar_hızı_1 = pd.qcut(veriSeti["ruzgar_hızı"], q = 5)
ruzgar_hızı_1.cat.categories
ruzgar_hızı_1.cat.rename_categories(["cok_dusuk_hız","dusuk_hız","ortalama_hız","yuksek_hız","cok_yuksek_hız"],inplace=True) 
ruzgar_hızı_1.value_counts()

#2
ruzgar_hızı_2 = pd.cut(veriSeti["ruzgar_hızı"], bins = 5)
ruzgar_hızı_2.cat.categories
ruzgar_hızı_2.cat.rename_categories(["cok_dusuk_hız","dusuk_hız","ortalama_hız","yuksek_hız","cok_yuksek_hız"],inplace=True) 
ruzgar_hızı_2.value_counts()

#3 
odev_bolme_kategorisi = ["0-2","3-4","5-6","7-8","9-10","11+"]
odev_bolme = [-1,2,4,6,8,10,20]
ruzgar_hızı_3 = pd.cut(veriSeti["ruzgar_hızı"] , bins= odev_bolme, labels=odev_bolme_kategorisi)
ruzgar_hızı_3.value_counts()


veriSeti.ruzgar_hızı.describe()
