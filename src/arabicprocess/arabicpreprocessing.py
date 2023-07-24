# -*- coding: utf-8 -*-

# Importing required libraries
import re
import nltk
import time
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Suppressing FutureWarning to keep the output clean
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppressing FutureWarning to keep the output clean
from string import digits
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Downloading NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')


# - - - - - - - - - - - - - -
# Aditional Stop words

data_into_list = ['،', 'ء', 'ءَ', 'آ', 'آب', 'آذار', 'آض', 'آل', 'آمينَ', 'آناء', 'آنفا', 'آه', 'آهاً', 'آهٍ', 'آهِ', 'أ', 'أبدا', 'أبريل', 'أبو', 'أبٌ', 'أجل', 'أجمع', 'أحد', 'أخبر', 'أخذ', 'أخو', 'أخٌ', 'أربع', 'أربعاء', 'أربعة', 'أربعمئة', 'أربعمائة', 'أرى', 'أسكن', 'أصبح', 'أصلا', 'أضحى', 'أطعم', 'أعطى', 'أعلم', 'أغسطس', 'أفريل', 'أفعل به', 'أفٍّ', 'أقبل', 'أكتوبر', 'أل', 'ألا', 'ألف', 'ألفى', 'أم', 'أما', 'أمام', 'أمامك', 'أمامكَ', 'أمد', 'أمس', 'أمسى', 'أمّا', 'أن', 'أنا', 'أنبأ', 'أنت', 'أنتم', 'أنتما', 'أنتن', 'أنتِ', 'أنشأ', 'أنه', 'أنًّ', 'أنّى', 'أهلا', 'أو', 'أوت', 'أوشك', 'أول', 'أولئك', 'أولاء', 'أولالك', 'أوّهْ', 'أى', 'أي', 'أيا', 'أيار', 'أيضا', 'أيلول', 'أين', 'أيّ', 'أيّان', 'أُفٍّ', 'ؤ', 'إحدى', 'إذ', 'إذا', 'إذاً', 'إذما', 'إذن', 'إزاء', 'إلى', 'إلي', 'إليكم', 'إليكما', 'إليكنّ', 'إليكَ', 'إلَيْكَ', 'إلّا', 'إمّا', 'إن', 'إنَّ', 'إى', 'إياك', 'إياكم', 'إياكما', 'إياكن', 'إيانا', 'إياه', 'إياها', 'إياهم', 'إياهما', 'إياهن', 'إياي', 'إيهٍ', 'ئ', 'ا', 'ا?', 'ا?ى', 'االا', 'االتى', 'ابتدأ', 'ابين', 'اتخذ', 'اثر', 'اثنا', 'اثنان', 'اثني', 'اثنين', 'اجل', 'احد', 'اخرى', 'اخلولق', 'اذا', 'اربعة', 'اربعون', 'اربعين', 'ارتدّ', 'استحال', 'اصبح', 'اضحى', 'اطار', 'اعادة', 'اعلنت', 'اف', 'اكثر', 'اكد', 'الآن', 'الألاء', 'الألى', 'الا', 'الاخيرة', 'الان', 'الاول', 'الاولى', 'التى', 'التي', 'الثاني', 'الثانية', 'الحالي', 'الذاتي', 'الذى', 'الذي', 'الذين', 'السابق', 'الف', 'اللاتي', 'اللتان', 'اللتيا', 'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'الماضي', 'المقبل', 'الوقت', 'الى', 'الي', 'اليه', 'اليها', 'اليوم', 'اما', 'امام', 'امس', 'امسى', 'ان', 'انبرى', 'انقلب', 'انه', 'انها', 'او', 'اول', 'اي', 'ايار', 'ايام', 'ايضا', 'ب', 'بؤسا', 'بإن', 'بئس', 'باء', 'بات', 'باسم', 'بان', 'بخٍ', 'بد', 'بدلا', 'برس', 'بسبب', 'بسّ', 'بشكل', 'بضع', 'بطآن', 'بعد', 'بعدا', 'بعض', 'بغتة', 'بل', 'بلى', 'بن', 'به', 'بها', 'بهذا', 'بيد', 'بين', 'بَسْ', 'بَلْهَ', 'ة', 'ت', 'تاء', 'تارة', 'تاسع', 'تانِ', 'تانِك', 'تبدّل', 'تجاه', 'تحت', 'تحوّل', 'تخذ', 'ترك', 'تسع', 'تسعة', 'تسعمئة', 'تسعمائة', 'تسعون', 'تسعين', 'تشرين', 'تعسا', 'تعلَّم', 'تفعلان', 'تفعلون', 'تفعلين', 'تكون', 'تلقاء', 'تلك', 'تم', 'تموز', 'تينك', 'تَيْنِ', 'تِه', 'تِي', 'ث', 'ثاء', 'ثالث', 'ثامن', 'ثان', 'ثاني', 'ثلاث', 'ثلاثاء', 'ثلاثة', 'ثلاثمئة', 'ثلاثمائة', 'ثلاثون', 'ثلاثين', 'ثم', 'ثمان', 'ثمانمئة', 'ثمانون', 'ثماني', 'ثمانية', 'ثمانين', 'ثمنمئة', 'ثمَّ', 'ثمّ', 'ثمّة', 'ج', 'جانفي', 'جدا', 'جعل', 'جلل', 'جمعة', 'جميع', 'جنيه', 'جوان', 'جويلية', 'جير', 'جيم', 'ح', 'حاء', 'حادي', 'حار', 'حاشا', 'حاليا', 'حاي', 'حبذا', 'حبيب', 'حتى', 'حجا', 'حدَث', 'حرى', 'حزيران', 'حسب', 'حقا', 'حمدا', 'حمو', 'حمٌ', 'حوالى', 'حول', 'حيث', 'حيثما', 'حين', 'حيَّ', 'حَذارِ', 'خ', 'خاء', 'خاصة', 'خال', 'خامس', 'خبَّر', 'خلا', 'خلافا', 'خلال', 'خلف', 'خمس', 'خمسة', 'خمسمئة', 'خمسمائة', 'خمسون', 'خمسين', 'خميس', 'د', 'دال', 'درهم', 'درى', 'دواليك', 'دولار', 'دون', 'دونك', 'ديسمبر', 'دينار', 'ذ', 'ذا', 'ذات', 'ذاك', 'ذال', 'ذانك', 'ذانِ', 'ذلك', 'ذهب', 'ذو', 'ذيت', 'ذينك', 'ذَيْنِ', 'ذِه', 'ذِي', 'ر', 'رأى', 'راء', 'رابع', 'راح', 'رجع', 'رزق', 'رويدك', 'ريال', 'ريث', 'رُبَّ', 'ز', 'زاي', 'زعم', 'زود', 'زيارة', 'س', 'ساء', 'سابع', 'سادس', 'سبت', 'سبتمبر', 'سبحان', 'سبع', 'سبعة', 'سبعمئة', 'سبعمائة', 'سبعون', 'سبعين', 'ست', 'ستة', 'ستكون', 'ستمئة', 'ستمائة', 'ستون', 'ستين', 'سحقا', 'سرا', 'سرعان', 'سقى', 'سمعا', 'سنة', 'سنتيم', 'سنوات', 'سوف', 'سوى', 'سين', 'ش', 'شباط', 'شبه', 'شتانَ', 'شخصا', 'شرع', 'شمال', 'شيكل', 'شين', 'شَتَّانَ', 'ص', 'صاد', 'صار', 'صباح', 'صبر', 'صبرا', 'صدقا', 'صراحة', 'صفر', 'صهٍ', 'صهْ', 'ض', 'ضاد', 'ضحوة', 'ضد', 'ضمن', 'ط', 'طاء', 'طاق', 'طالما', 'طرا', 'طفق', 'طَق', 'ظ', 'ظاء', 'ظل', 'ظلّ', 'ظنَّ', 'ع', 'عاد', 'عاشر', 'عام', 'عاما', 'عامة', 'عجبا', 'عدا', 'عدة', 'عدد', 'عدم', 'عدَّ', 'عسى', 'عشر', 'عشرة', 'عشرون', 'عشرين', 'عل', 'علق', 'علم', 'على', 'علي', 'عليك', 'عليه', 'عليها', 'علًّ', 'عن', 'عند', 'عندما', 'عنه', 'عنها', 'عوض', 'عيانا', 'عين', 'عَدَسْ', 'غ', 'غادر', 'غالبا', 'غدا', 'غداة', 'غير', 'غين', 'ـ', 'ف', 'فإن', 'فاء', 'فان', 'فانه', 'فبراير', 'فرادى', 'فضلا', 'فقد', 'فقط', 'فكان', 'فلان', 'فلس', 'فهو', 'فو', 'فوق', 'فى', 'في', 'فيفري', 'فيه', 'فيها', 'ق', 'قاطبة', 'قاف', 'قال', 'قام', 'قبل', 'قد', 'قرش', 'قطّ', 'قلما', 'قوة', 'ك', 'كأن', 'كأنّ', 'كأيّ', 'كأيّن', 'كاد', 'كاف', 'كان', 'كانت', 'كانون', 'كثيرا', 'كذا', 'كذلك', 'كرب', 'كسا', 'كل', 'كلتا', 'كلم', 'كلَّا', 'كلّما', 'كم', 'كما', 'كن', 'كى', 'كيت', 'كيف', 'كيفما', 'كِخ', 'ل', 'لأن', 'لا', 'لا سيما', 'لات', 'لازال', 'لاسيما', 'لام', 'لايزال', 'لبيك', 'لدن', 'لدى', 'لدي', 'لذلك', 'لعل', 'لعلَّ', 'لعمر', 'لقاء', 'لكن', 'لكنه', 'لكنَّ', 'للامم', 'لم', 'لما', 'لمّا', 'لن', 'له', 'لها', 'لهذا', 'لهم', 'لو', 'لوكالة', 'لولا', 'لوما', 'ليت', 'ليرة', 'ليس', 'ليسب', 'م', 'مئة', 'مئتان', 'ما', 'ما أفعله', 'ما انفك', 'ما برح', 'مائة', 'ماانفك', 'مابرح', 'مادام', 'ماذا', 'مارس', 'مازال', 'مافتئ', 'ماي', 'مايزال', 'مايو', 'متى', 'مثل', 'مذ', 'مرّة', 'مساء', 'مع', 'معاذ', 'معه', 'معها', 'مقابل', 'مكانكم', 'مكانكما', 'مكانكنّ', 'مكانَك', 'مليار', 'مليم', 'مليون', 'مما', 'من', 'منذ', 'منه', 'منها', 'مه', 'مهما', 'ميم', 'ن', 'نا', 'نبَّا', 'نحن', 'نحو', 'نعم', 'نفس', 'نفسه', 'نهاية', 'نوفمبر', 'نون', 'نيسان', 'نيف', 'نَخْ', 'نَّ', 'ه', 'هؤلاء', 'ها', 'هاء', 'هاكَ', 'هبّ', 'هذا', 'هذه', 'هل', 'هللة', 'هلم', 'هلّا', 'هم', 'هما', 'همزة', 'هن', 'هنا', 'هناك', 'هنالك', 'هو', 'هي', 'هيا', 'هيهات', 'هيّا', 'هَؤلاء', 'هَاتانِ', 'هَاتَيْنِ', 'هَاتِه', 'هَاتِي', 'هَجْ', 'هَذا', 'هَذانِ', 'هَذَيْنِ', 'هَذِه', 'هَذِي', 'هَيْهات', 'و', 'و6', 'وأبو', 'وأن', 'وا', 'واحد', 'واضاف', 'واضافت', 'واكد', 'والتي', 'والذي', 'وان', 'واهاً', 'واو', 'واوضح', 'وبين', 'وثي', 'وجد', 'وراءَك', 'ورد', 'وعلى', 'وفي', 'وقال', 'وقالت', 'وقد', 'وقف', 'وكان', 'وكانت', 'ولا', 'ولايزال', 'ولكن', 'ولم', 'وله', 'وليس', 'ومع', 'ومن', 'وهب', 'وهذا', 'وهو', 'وهي', 'وَيْ', 'وُشْكَانَ', 'ى', 'ي', 'ياء', 'يفعلان', 'يفعلون', 'يكون', 'يلي', 'يمكن', 'يمين', 'ين', 'يناير', 'يوان', 'يورو', 'يوليو', 'يوم', 'يونيو', 'ّأيّان', '']


# - - - - - - - - - - - - - -
# Arabic names

namess = ['ابتسام', 'ابتهاج', 'ابتهال', 'اجتهاد', 'ازدهار', 'اعتدال', 'اعتماد', 'افتخار', 'افتكار', 'البتول', 'البندري', 'الجازي', 'الجوري', 'الجوهرة', 'الريم', 'العنود', 'الهنوف', 'امتثال', 'امتياز', 'انبهاج', 'انتصار', 'انتظار', 'انسجام', 'انشراح', 'انشراف', 'آلاء', 'آمال', 'آمنة', 'آيات', 'آية', 'أبرار', 'أثير', 'أثيل', 'أحلام', 'أرجوان', 'أروى', 'أريام', 'أريج', 'أزهار', 'أسارير', 'أسرار', 'أسماء', 'أسمى', 'أسيل', 'أشجان', 'أطياف', 'أغاريد', 'أفراح', 'أفكار', 'أفنان', 'ألطاف', 'ألفت', 'أم كلثوم', 'أمارة', 'أمال', 'أمامة', 'أماني', 'أمجاد', 'أمل', 'أمنية', 'أمواج', 'أميرة', 'أميمة', 'أمينة', 'أنسام', 'أنغام', 'أنهار', 'أنوار', 'أنيسة', 'أوصاف', 'إباء', 'إجلال', 'إحسان', 'إخلاص', 'إدراك', 'إسراء', 'إسعاد', 'إصلاح', 'إعزاز', 'إقبال', 'إقدام', 'إكليل', 'إنصاف', 'إنعام', 'إيثار', 'إيمان', 'إيناس', 'بائقة', 'بادرة', 'بارزة', 'بارعة', 'باركة', 'بارية', 'بازعة', 'باسطة', 'باسقة', 'باسلة', 'بانة', 'بتول', 'بثينة', 'بدرية', 'بديعة', 'براء', 'بسمة', 'بسيمة', 'بشاير', 'بشرى', 'بشيرة', 'بلقيس', 'بليغة', 'بهيجة', 'بهيسة', 'بيان', 'تارا', 'تالا', 'تانيا', 'تغاني', 'تمارة', 'تماضر', 'تماني', 'تهاني', 'تودد', 'توليب', 'تولين', 'تيسير', 'تيماء', 'ثراء', 'ثناء', 'ثوبة', 'ثويبة', 'جبرة', 'جبلة', 'جمانة', 'جميلة', 'جنا', 'جنى', 'جوان', 'جواهر', 'جود', 'جوري', 'جوهرة', 'جوى', 'جويرية', 'جيداء', 'جيهان', 'حسناء', 'حسنية', 'حصة', 'حفصة', 'حكمت', 'حليمة', 'حنان', 'حنيفة', 'حور', 'حوراء', 'حورية', 'حياة', 'خاتمة', 'خالدة', 'خديجة', 'خزامى', 'خزنة', 'خزينة', 'خلود', 'خولة', 'دارين', 'داليا', 'دانا', 'دانة', 'دانيا', 'دانية', 'دعجاء', 'دلال', 'دنيا', 'ديم', 'ديما', 'ديمة', 'دينا', 'ذبيانة', 'ذكية', 'رابعة', 'راتبة', 'راجحة', 'راحيل', 'راسخة', 'راغبة', 'رافقة', 'رافية', 'راقية', 'راما', 'رامة', 'رانيا', 'ربى', 'رجاء', 'رجيحة', 'رحاب', 'رحمة', 'رحيق', 'رخاء', 'ردينة', 'رزان', 'رزينة', 'رسمية', 'رسيل', 'رسيمة', 'رشا', 'رشيقة', 'رصينة', 'رضا', 'رضوة', 'رضوى', 'رغد', 'رفعة', 'رفيدة', 'رفيعة', 'رفيف', 'رقية', 'رمزية', 'رنا', 'رند', 'رنيم', 'رهام', 'رهف', 'رهيفة', 'رواء', 'روابي', 'روان', 'روز', 'روضة', 'رولا', 'رؤى', 'ريا', 'ريتا', 'ريتاج', 'ريتال', 'ريحانة', 'ريفان', 'ريم', 'ريما', 'ريماس', 'ريمة', 'ريناد', 'ريهام', 'ريوف', 'زاهية', 'زبيدة', 'زكية', 'زهراء', 'زهرة', 'زهور', 'زهية', 'زينب', 'زينة', 'سؤدد', 'سارا', 'سارة', 'سالي', 'سامية', 'ساندرا', 'ساهدة', 'ساهرة', 'سبيعة', 'سجى', 'سحر', 'سدر', 'سدن', 'سديم', 'سدين', 'سعاد', 'سعدة', 'سعيدة', 'سفيانة', 'سكينة', 'سلامة', 'سلطانة', 'سلمى', 'سلوى', 'سمة', 'سمر', 'سمراء', 'سمية', 'سميحة', 'سميرة', 'سناء', 'سندس', 'سها', 'سهاد', 'سهام', 'سهيلة', 'سونيا', 'سيرين', 'سيلين', 'شادن', 'شاكرة', 'شاهدة', 'شاهرة', 'شجن', 'شجون', 'شذى', 'شروق', 'شريفة', 'شكرية', 'شموخ', 'شهد', 'شهلاء', 'شوق', 'شيلاء', 'شيماء', 'شيهانة', 'صائنة', 'صابرة', 'صبا', 'صفا', 'صفاء', 'صفية', 'صمود', 'ضياء', 'طامحة', 'طريفة', 'طيب', 'طيف', 'ظافرة', 'ظبية', 'عايدة', 'عائدة', 'عائذة', 'عائشة', 'عابدة', 'عاتكة', 'عاطفة', 'عالية', 'عبادة', 'عبلة', 'عبير', 'عذاري', 'عذبة', 'عزة', 'عزيزة', 'عفاف', 'علياء', 'عنود', 'عهد', 'عهود', 'عيدة', 'غادة', 'غرام', 'غريبة', 'غزل', 'غزيل', 'غصون', 'غلا', 'غناء', 'غيثة', 'غيد', 'غيداء', 'فائقة', 'فاتن', 'فادية', 'فاطمة', 'فتحية', 'فتون', 'فتيحة', 'فجر', 'فخرية', 'فدوى', 'فرح', 'فردوس', 'فريدة', 'فصيحة', 'فكرية', 'فلوة', 'فنن', 'فهدة', 'فهيمة', 'فوزية', 'فيروز', 'قمر', 'قنوت', 'كاتيا', 'كادي', 'كاظمة', 'كاملة', 'كبرى', 'كريمة', 'كفى', 'كندة', 'كنزى', 'كوثر', 'كيان', 'لوزت', 'لولوة', 'لؤلؤة', 'لارا', 'لانا', 'لبانة', 'لبنى', 'لجين', 'لجينة', 'لذة', 'لطفية', 'لطيفة', 'لمى', 'لمياء', 'لميس', 'لهيفة', 'ليال', 'ليالي', 'ليان', 'ليلى', 'لين', 'لينا', 'ماريا', 'مارية', 'ماهة', 'ماهرة', 'مبروكة', 'متام', 'مثابة', 'مجاهدة', 'مجيبة', 'مجيدة', 'مخلصة', 'مدركة', 'مديدة', 'مرام', 'مرح', 'مرهفة', 'مروة', 'مروى', 'مريم', 'مزنة', 'مزينة', 'مساعدة', 'مسفرة', 'مسك', 'مشاعل', 'مصلحة', 'مضاوي', 'مطاعة', 'مطيعة', 'معصومة', 'معطية', 'معينة', 'مغيثة', 'مقبلة', 'ملهمة', 'ممدوحة', 'منار', 'مناصف', 'منال', 'مناير', 'منتهى', 'منذرة', 'منصفة', 'منى', 'منيبة', 'منيرة', 'مها', 'مهرة', 'مهيبة', 'موزة', 'موهبة', 'مي', 'ميا', 'ميادة', 'ميار', 'ميثاء', 'ميرا', 'ميس', 'ميساء', 'ميسون', 'ميمونة', 'ناجحة', 'ناجدة', 'نادية', 'نادين', 'ناشئة', 'ناصحة', 'ناهد', 'ناهية', 'نبيلة', 'نبيهة', 'نجد', 'نجلاء', 'نجوانة', 'نجود', 'نجيبة', 'نجيحة', 'نداء', 'ندى', 'نرجس', 'نزيهة', 'نسرين', 'نسيم', 'نشوة', 'نشوى', 'نصر', 'نضال', 'نظيرة', 'نعمة', 'نعيم', 'نهاد', 'نهال', 'نهلة', 'نهى', 'نوار', 'نوال', 'نور', 'نورا', 'نورة', 'نورس', 'نورهان', 'نوف', 'هائلة', 'هاجر', 'هالة', 'هانئة', 'هبة', 'هدى', 'هدير', 'هديل', 'هناء', 'هنادي', 'هند', 'هنوف', 'هوازن', 'هيام', 'هيفاء', 'وتين', 'وجدان', 'ود', 'وديعة', 'ورد', 'وردة', 'وسام', 'وسن', 'وسيمة', 'وضحى', 'وعد', 'وفاء', 'وفيقة', 'ولاء', 'ولهانة', 'وليفة', 'وميض', 'وهبة', 'يارا', 'ياسمين', 'يسرا', 'يسرى', 'يمن', 'يمنى', 'ابراهيم', 'اسلم', 'البراء', 'الحبيب', 'الخضر', 'العابدين', 'المثنى', 'آدم', 'أبان', 'أبلج', 'أبو بكر', 'أجاويد', 'أجيد', 'أحمد', 'أحنف', 'أخزم', 'أخضر', 'أخطب', 'أدعج', 'أدغم', 'أدهم', 'أديب', 'أرغد', 'أرقم', 'أريب', 'أزهر', 'أزور', 'أسامة', 'أسد', 'أسعد', 'أسلم', 'أسمر', 'أشجع', 'أشرف', 'أشرم', 'أشقر', 'أشهب', 'أشهم', 'أشيم', 'أصيل', 'أعسر', 'أغلب', 'أغيد', 'أكبر', 'أكثم', 'أكرم', 'أكمل', 'أمان', 'أمجد', 'أمية', 'أمير', 'أمين', 'أنس', 'أنعم', 'أنمار', 'أنور', 'أهيم', 'أوس', 'أويس', 'أيسر', 'أيمن', 'أيهم', 'أيوب', 'إبراهيم', 'إحسان', 'إدريس', 'إسحاق', 'إسلام', 'إسماعيل', 'إمام', 'إياد', 'إياس', 'إيهاب', 'باتل', 'بادي', 'بارح', 'بارع', 'باسط', 'باسق', 'باسل', 'باسم', 'باشق', 'باهر', 'بدر', 'بديع', 'براء', 'برد', 'برهان', 'بسام', 'بسيم', 'بشار', 'بشارة', 'بشر', 'بشير', 'بطاح', 'بطرس', 'بكار', 'بكر', 'بكري', 'بلال', 'بليغ', 'بندر', 'بهاء', 'بهجت', 'بهلول', 'بهيج', 'تائب', 'تامر', 'تركي', 'تغلب', 'تقي', 'تليد', 'تمام', 'تميم', 'تواب', 'توفيق', 'تيسير', 'تيم', 'ثائب', 'ثائر', 'ثابت', 'ثروت', 'ثنيان', 'ثواب', 'جابر', 'جازم', 'جاسم', 'جامع', 'جبر', 'جبران', 'جبل', 'جدعان', 'جراح', 'جرير', 'جساس', 'جعفر', 'جلال', 'جليل', 'جمال', 'جمعان', 'جمعة', 'جميل', 'جنادة', 'جندل', 'جهاد', 'جواد', 'جودة', 'جودت', 'حاتم', 'حارب', 'حارث', 'حازم', 'حاشد', 'حافظ', 'حامد', 'حجاب', 'حجاج', 'حجر', 'حزام', 'حزم', 'حسام', 'حسان', 'حسن', 'حسني', 'حسنين', 'حسون', 'حسين', 'حصيف', 'حفص', 'حكيم', 'حماد', 'حمد', 'حمدان', 'حمدون', 'حمزة', 'حميد', 'حنبل', 'حنشل', 'حنظلة', 'حواس', 'حيدر', 'خاتم', 'خازن', 'خالد', 'خالص', 'خباب', 'خزعل', 'خضر', 'خطاب', 'خلدون', 'خلف', 'خليفة', 'خليل', 'خليوي', 'خميس', 'خويلد', 'داني', 'داود', 'داوود', 'دحام', 'درغام', 'دريد', 'دعيج', 'دغيم', 'دهيم', 'ذاكر', 'ذباح', 'ذبيان', 'ذعار', 'ذياب', 'رأفت', 'رؤوف', 'رئيف', 'رائد', 'رائض', 'رائف', 'رابح', 'راتب', 'راجح', 'راجي', 'راسخ', 'راسم', 'راشد', 'راضي', 'راغب', 'راغد', 'رافد', 'رافع', 'رافق', 'رافي', 'راكان', 'رامز', 'رامي', 'رباح', 'ربحي', 'ربيح', 'ربيع', 'رجاء', 'رجب', 'رحال', 'رديف', 'رزق', 'رزين', 'رسمي', 'رسول', 'رسيم', 'رشاد', 'رشوان', 'رشيد', 'رشيق', 'رصين', 'رضا', 'رضوان', 'رضي', 'رعد', 'رغال', 'رغيب', 'رغيد', 'رفاعة', 'رفاعي', 'رفعت', 'رفيع', 'رفيق', 'ركان', 'ركين', 'رماح', 'رمزي', 'رمضان', 'رهيف', 'رويشد', 'رياض', 'ريان', 'زاخر', 'زاكي', 'زاهد', 'زاهر', 'زاهي', 'زايد', 'زبير', 'زغلول', 'زكريا', 'زهير', 'زياد', 'زيد', 'زيدان', 'زيدون', 'زين', 'ساجد', 'ساجر', 'ساجع', 'ساجي', 'سادن', 'سارح', 'ساري', 'ساطع', 'ساعد', 'ساعف', 'سالم', 'سامح', 'سامر', 'سامي', 'ساهد', 'ساهر', 'سبع', 'ستار', 'سحيم', 'سداد', 'سراج', 'سرمد', 'سرور', 'سطام', 'سعد', 'سعدون', 'سعود', 'سعيد', 'سفيان', 'سلام', 'سلامة', 'سلطان', 'سلمان', 'سلمة', 'سليم', 'سمعان', 'سمعون', 'سميح', 'سمير', 'سميع', 'سنان', 'سهل', 'سهيل', 'سويد', 'سيار', 'سيد', 'سيف', 'شادي', 'شافع', 'شافي', 'شاكر', 'شامخ', 'شاهد', 'شاهر', 'شبل', 'شبيب', 'شجاع', 'شداد', 'شديد', 'شريح', 'شعبان', 'شعيب', 'شفيع', 'شفيق', 'شكري', 'شكور', 'شماخ', 'شمر', 'شمس الدين', 'شهاب', 'شهم', 'شهيد', 'شوقي', 'شوكت', 'شيبان', 'شيبة', 'صائن', 'صابر', 'صادح', 'صادر', 'صارم', 'صاعد', 'صافي', 'صباح', 'صبيح', 'صدام', 'صديق', 'صفوان', 'صفوت', 'صقر', 'صلاح', 'صلاح الدين', 'صمد', 'صمصام', 'صميم', 'صهيب', 'ضاري', 'ضاهر', 'ضاوي', 'ضرار', 'ضرام', 'ضياء', 'ضيغم', 'ضيف', 'طارق', 'طامح', 'طاهر', 'طريف', 'طلال', 'طموح', 'ظافر', 'ظاهر', 'ظفار', 'ظفير', 'عايض', 'عائد', 'عائذ', 'عائض', 'عابد', 'عادل', 'عارف', 'عازم', 'عاطف', 'عاكف', 'عامر', 'عباد', 'عبادة', 'عباس', 'عبدالباسط', 'عبدالباقي', 'عبدالبصير', 'عبدالجبار', 'عبدالجليل', 'عبدالجميل', 'عبدالحفيظ', 'عبدالحق', 'عبدالحكيم', 'عبدالحليم', 'عبدالحميد', 'عبدالحي', 'عبدالخالق', 'عبدالرؤوف', 'عبدالرب', 'عبدالرحمن', 'عبدالرحيم', 'عبدالرزاق', 'عبدالرشيد', 'عبدالرقيب', 'عبدالسلام', 'عبدالشكور', 'عبدالصمد', 'عبدالظاهر', 'عبدالعزيز', 'عبدالعظيم', 'عبدالعليم', 'عبدالغفور', 'عبدالغني', 'عبدالفتاح', 'عبدالقادر', 'عبدالقدوس', 'عبدالقيوم', 'عبدالكريم', 'عبداللطيف', 'عبدالله', 'عبدالمجيد', 'عبدالمطلب', 'عبدالملك', 'عبدالمنان', 'عبدالمنعم', 'عبدالنور', 'عبدالهادي', 'عبدالواحد', 'عبدالودود', 'عبدالوهاب', 'عبدربه', 'عبده', 'عبدو', 'عبدي', 'عبودة', 'عبيد', 'عبيدة', 'عتاب', 'عتابة', 'عثمان', 'عجاج', 'عجرم', 'عدلي', 'عدنان', 'عدوان', 'عدي', 'عديل', 'عراب', 'عراد', 'عرفات', 'عرفة', 'عروة', 'عريب', 'عز', 'عزب', 'عزت', 'عزيز', 'عشير', 'عصام', 'عصمت', 'عطا', 'عطاء', 'عفان', 'عفيف', 'عقاب', 'عقال', 'عقبة', 'عقل', 'عقلة', 'عقيل', 'عكاشة', 'عكرمة', 'عكلة', 'علاء', 'علاء الدين', 'علوان', 'علي', 'عليان', 'عليم', 'عماد', 'عمار', 'عمارة', 'عمر', 'عمران', 'عمرو', 'عميد', 'عمير', 'عنبر', 'عنتر', 'عنترة', 'عواد', 'عوض', 'عوف', 'عون', 'عوني', 'عويس', 'عياش', 'عيد', 'عيسى', 'عيضة', 'غازي', 'غافر', 'غالب', 'غالي', 'غانم', 'غريب', 'غزوان', 'غسان', 'غضنفر', 'غفار', 'غلاب', 'غلام', 'غنيم', 'غوار', 'غوث', 'غياث', 'غيث', 'فؤاد', 'فائز', 'فائق', 'فاتح', 'فاخر', 'فادي', 'فارس', 'فارض', 'فارع', 'فاروق', 'فاضل', 'فالح', 'فتاح', 'فتح', 'فتحي', 'فتوح', 'فتيح', 'فخر', 'فخري', 'فراج', 'فراس', 'فرج', 'فرحات', 'فرحان', 'فرناس', 'فريج', 'فريد', 'فضل', 'فضيل', 'فطين', 'فكري', 'فلوح', 'فليح', 'فهد', 'فهمي', 'فهيد', 'فهيم', 'فواز', 'فوزي', 'فياض', 'فيصل', 'فيض', 'فيضي', 'قائد', 'قابس', 'قابوس', 'قادر', 'قاسم', 'قاهر', 'قتادة', 'قتيبة', 'قثم', 'قداح', 'قدامة', 'قدري', 'قسام', 'قسيم', 'قصي', 'قطامي', 'قطب', 'قعقاع', 'قيس', 'كاسب', 'كاسر', 'كاظم', 'كامل', 'كرم', 'كريم', 'كعب', 'كلاب', 'كليب', 'كليم', 'كمال', 'كميت', 'كنان', 'كنعان', 'كهلان', 'لؤي', 'لباب', 'لبيب', 'لبيد', 'لطفي', 'لطيف', 'لقمان', 'لماح', 'لمعي', 'ليث', 'مأمون', 'مؤيد', 'ماجد', 'مارسيل', 'مازن', 'ماضي', 'مالك', 'ماهر', 'مبارك', 'مبخوت', 'مبرور', 'مبروك', 'متعب', 'مثاب', 'مجاهد', 'مجاور', 'مجتبى', 'مجدي', 'مجيب', 'مجيد', 'مجير', 'محبوب', 'محجن', 'محرز', 'محرم', 'محسن', 'محمد', 'محمود', 'محيا', 'مخلد', 'مخلص', 'مخلف', 'مدثر', 'مدحة', 'مدحت', 'مدرك', 'مدلج', 'مديد', 'مراد', 'مرتجى', 'مرتضى', 'مرحب', 'مرداس', 'مرزوق', 'مرسي', 'مرشد', 'مرشدي', 'مروان', 'مزاحم', 'مزن', 'مزهر', 'مساعد', 'مستور', 'مسعود', 'مسفر', 'مسلم', 'مسلمة', 'مسيب', 'مشاري', 'مشتاق', 'مشرف', 'مشعل', 'مصباح', 'مصطفى', 'مصعب', 'مصلح', 'مضر', 'مطاع', 'مطاوع', 'مطر', 'مطلب', 'مطلق', 'مطير', 'مطيع', 'مظفر', 'مظهر', 'معاد', 'معاذ', 'معاوية', 'معتصم', 'معتضد', 'معتوق', 'معد', 'معروف', 'معصوم', 'معطي', 'معلوف', 'معمر', 'معن', 'معين', 'معين الدين', 'مغوار', 'مغيث', 'مغيرة', 'مفتاح', 'مقبل', 'مقتدي', 'مقرن', 'مكتوم', 'مكين', 'ملحم', 'ملهم', 'ممدوح', 'مناحي', 'مناع', 'مناف', 'منان', 'منتصر', 'منتظر', 'منذر', 'منصف', 'منصور', 'منعم', 'منقذ', 'منهل', 'منيب', 'منير', 'منيع', 'مهاب', 'مهدي', 'مهند', 'مهيب', 'مهيوب', 'موفق', 'ميثم', 'ميلاد', 'ميمون', 'نائل', 'ناجي', 'نادر', 'ناشئ', 'ناشد', 'ناشر', 'ناصح', 'ناصر', 'ناصر الدين', 'ناصف', 'ناصيف', 'ناظم', 'نافع', 'نامي', 'ناهض', 'ناهل', 'ناهي', 'نايف', 'نبال', 'نبيل', 'نبيه', 'نجاد', 'نجدة', 'نجدت', 'نجم', 'نجم الدين', 'نجيب', 'نديم', 'نذير', 'نزار', 'نزيه', 'نسيب', 'نسيم', 'نشأت', 'نشمي', 'نصار', 'نصر', 'نصرة', 'نصرت', 'نصوح', 'نصير', 'نصيف', 'نضال', 'نضير', 'نظمي', 'نظير', 'نعمان', 'نعمت', 'نعيم', 'نمر', 'نهيان', 'نوار', 'نواس', 'نواف', 'نوح', 'نورس', 'نوري', 'نوفل', 'هائل', 'هاجد', 'هادي', 'هاشم', 'هاني', 'هانئ', 'هبيرة', 'هجرس', 'هريرة', 'هزاع', 'هشام', 'هلال', 'همام', 'هواري', 'هود', 'هيثم', 'وائل', 'وارف', 'وديع', 'ورد', 'وسام', 'وسيم', 'وفيق', 'ولاء الدين', 'وليد', 'وليف', 'وهاب', 'وهب', 'وهبة', 'ياسر', 'يحيى', 'يزيد', 'يعقوب']


# - - - - - - - - - - - - - -

# Steamming

SPACE            = u'\u0020'
EXCLAMATION      = u'\u0021'
en_QUOTATION     = u'\u0022'
NUMBER_SIGN      = u'\u0023'
DOLLAR_SIGN      = u'\u0024'
en_PERCENT       = u'\u0025' #unichr(37)
AMPERSAND        = u'\u0026'
APOSTROPHE       = u'\u0027' #unichr(39)
LEFT_PARENTHESIS = u'\u0028'
RIGHT_PARENTHESIS= u'\u0029'
ASTERISK         = u'\u002a' #unichr(42)
PLUS_SIGN        = u'\u002b'
en_COMMA         = u'\u002c'
HYPHEN_MINUS     = u'\u002d' #unichr(45)
en_FULL_STOP     = u'\u002e'
SLASH            = u'\u002f'
# English Numbers
ZERO             = u'\u0030'
ONE              = u'\u0031'
TWO              = u'\u0032'
THREE            = u'\u0033'
FOUR             = u'\u0034'
FIVE             = u'\u0035'
SIX              = u'\u0036'
SEVEN            = u'\u0037'
EIGHT            = u'\u0038'
NINE             = u'\u0039'

en_COLON             = u'\u003a' #unichr(58)
en_SEMICOLON         = u'\u003b'
en_LESS_THAN         = u'\u003c' #unichr(60)
en_EQUALS_SIGN       = u'\u003d' #
en_GREATER_THAN      = u'\u003e' #
en_QUESTION          = u'\u003f'
COMMERCIAL_AT        = u'\u0040'

LEFT_SQUARE_BRACKET  = u'\u005b'
BACKSLASH            = u'\u005c'
RIGHT_SQUARE_BRACKET = u'\u005d'
CIRCUMFLEX_ACCENT    = u'\u005e'
UNDERSCORE           = u'\u005f'
GRAVE_ACCENT         = u'\u0060'

LEFT_CURLY_BRACKET   = u'\u007b'
VERTICAL_LINE        = u'\u007c'
RIGHT_CURLY_BRACKET  = u'\u007d'
TILDE                = u'\u007e'

Leftpointing_double_angle_quotation_mark  = u'\u00ab'
MIDDLE_DOT                                = u'\u00b7' #unichr(183)
Rightpointing_double_angle_quotation_mark = u'\u00bb'

ar_COMMA         = u'\u060c'
ar_DATE_SEPARATO = u'\u060d'
ar_SEMICOLON     = u'\u061b'
ar_QUESTION      = u'\u061f' #QUESTION
HAMZA            = u'\u0621'
ALEF_MADDA       = u'\u0622'
ALEF_HAMZA_ABOVE = u'\u0623'
WAW_HAMZA        = u'\u0624'
ALEF_HAMZA_BELOW = u'\u0625'
YEH_HAMZA        = u'\u0626'
ALEF             = u'\u0627'
BEH              = u'\u0628'
TEH_MARBUTA      = u'\u0629'
TEH              = u'\u062a'
THEH             = u'\u062b'
JEEM             = u'\u062c'
HAH              = u'\u062d'
KHAH             = u'\u062e'
DAL              = u'\u062f'
THAL             = u'\u0630'
REH              = u'\u0631'
ZAIN             = u'\u0632'
SEEN             = u'\u0633'
SHEEN            = u'\u0634'
SAD              = u'\u0635'
DAD              = u'\u0636'
TAH              = u'\u0637'
ZAH              = u'\u0638'
AIN              = u'\u0639'
GHAIN            = u'\u063a'
TATWEEL          = u'\u0640'
FEH              = u'\u0641'
QAF              = u'\u0642'
KAF              = u'\u0643'
LAM              = u'\u0644'
MEEM             = u'\u0645'
NOON             = u'\u0646'
HEH              = u'\u0647'
WAW              = u'\u0648'
ALEF_MAKSURA     = u'\u0649'
YEH              = u'\u064a'
MADDA_ABOVE      = u'\u0653'
HAMZA_ABOVE      = u'\u0654'
HAMZA_BELOW      = u'\u0655'

ar_ZERO          = u'\u0660'
ar_ONE           = u'\u0661'
ar_TWO           = u'\u0662'
ar_THREE         = u'\u0663'
ar_FOUR          = u'\u0664'
ar_FIVE          = u'\u0665'
ar_SIX           = u'\u0666'
ar_SEVEN         = u'\u0667'
ar_EIGHT         = u'\u0668'
ar_NINE          = u'\u0669'

ar_PERCENT       = u'\u066a' #PERCENT
ar_DECIMAL       = u'\u066b'
ar_THOUSANDS     = u'\u066c'
ar_STAR          = u'\u066d'
MINI_ALEF        = u'\u0670'
ALEF_WASLA       = u'\u0671'
ar_FULL_STOP     = u'\u06d4' #FULL_STOP
BYTE_ORDER_MARK  = u'\ufeff'

# Diacritics
FATHATAN         = u'\u064b'
DAMMATAN         = u'\u064c'
KASRATAN         = u'\u064d'
FATHA            = u'\u064e'
DAMMA            = u'\u064f'
KASRA            = u'\u0650'
SHADDA           = u'\u0651'
SUKUN            = u'\u0652'

# Small Letters
SMALL_ALEF      =u"\u0670"
SMALL_WAW       =u"\u06E5"
SMALL_YEH       =u"\u06E6"
#Ligatures
LAM_ALEF                    =u'\ufefb'
LAM_ALEF_HAMZA_ABOVE        =u'\ufef7'
LAM_ALEF_HAMZA_BELOW        =u'\ufef9'
LAM_ALEF_MADDA_ABOVE        =u'\ufef5'
simple_LAM_ALEF             =u'\u0644\u0627'
simple_LAM_ALEF_HAMZA_ABOVE =u'\u0644\u0623'
simple_LAM_ALEF_HAMZA_BELOW =u'\u0644\u0625'
simple_LAM_ALEF_MADDA_ABOVE =u'\u0644\u0622'

Left_double_quotation_mark  = u'\u201c' #unichr(8220)
Right_double_quotation_mark = u'\u201d' #unichr(8221)
BULLET                      = u'\u2022'

# groups
LETTERS=u''.join([
        ALEF , BEH , TEH  , TEH_MARBUTA  , THEH  , JEEM  , HAH , KHAH ,
        DAL   , THAL  , REH   , ZAIN  , SEEN   , SHEEN  , SAD , DAD , TAH   , ZAH   ,
        AIN   , GHAIN   , FEH  , QAF , KAF , LAM , MEEM , NOON, HEH , WAW, YEH  ,
        HAMZA  ,  ALEF_MADDA , ALEF_HAMZA_ABOVE , WAW_HAMZA   , ALEF_HAMZA_BELOW  , YEH_HAMZA  ,
        ])

TASHKEEL =(FATHATAN, DAMMATAN, KASRATAN,
            FATHA,DAMMA,KASRA,
            SUKUN,
            SHADDA);
HARAKAT =(  FATHATAN,   DAMMATAN,   KASRATAN,
            FATHA,  DAMMA,  KASRA,
            SUKUN
            );
SHORTHARAKAT =( FATHA,  DAMMA,  KASRA, SUKUN);

TANWIN =(FATHATAN,  DAMMATAN,   KASRATAN);


LIGUATURES=(
            LAM_ALEF,
            LAM_ALEF_HAMZA_ABOVE,
            LAM_ALEF_HAMZA_BELOW,
            LAM_ALEF_MADDA_ABOVE,
            );
HAMZAT=(
            HAMZA,
            WAW_HAMZA,
            YEH_HAMZA,
            HAMZA_ABOVE,
            HAMZA_BELOW,
            ALEF_HAMZA_BELOW,
            ALEF_HAMZA_ABOVE,
            );
ALEFAT=(
            ALEF,
            ALEF_MADDA,
            ALEF_HAMZA_ABOVE,
            ALEF_HAMZA_BELOW,
            ALEF_WASLA,
            ALEF_MAKSURA,
            SMALL_ALEF,

        );
WEAK   = ( ALEF, WAW, YEH, ALEF_MAKSURA);
YEHLIKE= ( YEH,  YEH_HAMZA,  ALEF_MAKSURA,   SMALL_YEH  );

WAWLIKE     =   ( WAW,  WAW_HAMZA,  SMALL_WAW );
TEHLIKE     =   ( TEH,  TEH_MARBUTA );

SMALL   =( SMALL_ALEF, SMALL_WAW, SMALL_YEH)
MOON =(HAMZA            ,
        ALEF_MADDA       ,
        ALEF_HAMZA_ABOVE ,
        ALEF_HAMZA_BELOW ,
        ALEF             ,
        BEH              ,
        JEEM             ,
        HAH              ,
        KHAH             ,
        AIN              ,
        GHAIN            ,
        FEH              ,
        QAF              ,
        KAF              ,
        MEEM             ,
        HEH              ,
        WAW              ,
        YEH
    );
SUN=(
        TEH              ,
        THEH             ,
        DAL              ,
        THAL             ,
        REH              ,
        ZAIN             ,
        SEEN             ,
        SHEEN            ,
        SAD              ,
        DAD              ,
        TAH              ,
        ZAH              ,
        LAM              ,
        NOON             ,
    );
AlphabeticOrder={
                ALEF             : 1,
                BEH              : 2,
                TEH              : 3,
                TEH_MARBUTA      : 3,
                THEH             : 4,
                JEEM             : 5,
                HAH              : 6,
                KHAH             : 7,
                DAL              : 8,
                THAL             : 9,
                REH              : 10,
                ZAIN             : 11,
                SEEN             : 12,
                SHEEN            : 13,
                SAD              : 14,
                DAD              : 15,
                TAH              : 16,
                ZAH              : 17,
                AIN              : 18,
                GHAIN            : 19,
                FEH              : 20,
                QAF              : 21,
                KAF              : 22,
                LAM              : 23,
                MEEM             : 24,
                NOON             : 25,
                HEH              : 26,
                WAW              : 27,
                YEH              : 28,
                HAMZA            : 29,

                ALEF_MADDA       : 29,
                ALEF_HAMZA_ABOVE : 29,
                WAW_HAMZA        : 29,
                ALEF_HAMZA_BELOW : 29,
                YEH_HAMZA        : 29,
                }
NAMES ={
                ALEF             :  u"ألف",
                BEH              : u"باء",
                TEH              : u'تاء' ,
                TEH_MARBUTA      : u'تاء مربوطة' ,
                THEH             : u'ثاء' ,
                JEEM             : u'جيم' ,
                HAH              : u'حاء' ,
                KHAH             : u'خاء' ,
                DAL              : u'دال' ,
                THAL             : u'ذال' ,
                REH              : u'راء' ,
                ZAIN             : u'زاي' ,
                SEEN             : u'سين' ,
                SHEEN            : u'شين' ,
                SAD              : u'صاد' ,
                DAD              : u'ضاد' ,
                TAH              : u'طاء' ,
                ZAH              : u'ظاء' ,
                AIN              : u'عين' ,
                GHAIN            : u'غين' ,
                FEH              : u'فاء' ,
                QAF              : u'قاف' ,
                KAF              : u'كاف' ,
                LAM              : u'لام' ,
                MEEM             : u'ميم' ,
                NOON             : u'نون' ,
                HEH              : u'هاء' ,
                WAW              : u'واو' ,
                YEH              : u'ياء' ,
                HAMZA            : u'همزة' ,

                TATWEEL          : u'تطويل' ,
                ALEF_MADDA       : u'ألف ممدودة' ,
                ALEF_MAKSURA      : u'ألف مقصورة' ,
                ALEF_HAMZA_ABOVE : u'همزة على الألف' ,
                WAW_HAMZA        : u'همزة على الواو' ,
                ALEF_HAMZA_BELOW : u'همزة تحت الألف' ,
                YEH_HAMZA        : u'همزة على الياء' ,
                FATHATAN         : u'فتحتان',
                DAMMATAN         : u'ضمتان',
                KASRATAN         : u'كسرتان',
                FATHA            : u'فتحة',
                DAMMA            : u'ضمة',
                KASRA            : u'كسرة',
                SHADDA           : u'شدة',
                SUKUN            : u'سكون',
                }
#!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
PUNCTUATIONS = (ar_COMMA, ar_SEMICOLON, ar_QUESTION, ar_PERCENT, ar_DECIMAL,
                ar_THOUSANDS, ar_FULL_STOP,

                EXCLAMATION, en_QUOTATION, NUMBER_SIGN, DOLLAR_SIGN, en_PERCENT,
                AMPERSAND, LEFT_PARENTHESIS, RIGHT_PARENTHESIS,
                ASTERISK, PLUS_SIGN, en_COMMA, HYPHEN_MINUS, en_FULL_STOP,
                SLASH, en_COLON, en_SEMICOLON, en_LESS_THAN, en_EQUALS_SIGN,
                en_GREATER_THAN, en_QUESTION, COMMERCIAL_AT, LEFT_SQUARE_BRACKET,
                BACKSLASH, RIGHT_SQUARE_BRACKET, CIRCUMFLEX_ACCENT, UNDERSCORE,
                GRAVE_ACCENT, LEFT_CURLY_BRACKET, VERTICAL_LINE,
                RIGHT_CURLY_BRACKET, TILDE, Leftpointing_double_angle_quotation_mark,
                MIDDLE_DOT, Rightpointing_double_angle_quotation_mark ) #APOSTROPHE

larkey_defarticles = (u"ال", u"وال", u"بال", u"كال", u"فال", u"لل")
larkey_suffixes = (u"ها", u"ان", u"ات", u"ون", u"ين", u"يه", u"ية", u"ه", u"ة", u"ي")




#======================================= CLASS BEGAIN HERE =======================================

class process():

  def __init__(self, process_list=['clean','st_remove','stemm'], details=True, chart=True, csv=True, column_stay=True):
        self.process_list=process_list
        self.details=details
        self.chart=chart
        self.csv=csv
        self.column_stay=column_stay



  #=============================== ORIGINAL ===============================

  # Get Tokens (words) count
  def get_token_count(self, text):
      tokens = re.findall(r'\b\w+\b|\S', text)
      return len(tokens)

  # Get all charecters count
  def get_char_count(self, text):
    return len(text)



  #============================= Data Cleaning ============================

  def clean_texts(self, text):

    remove_digits = str.maketrans('', '', digits)
    text =text.translate(remove_digits)
    text=text.translate(str.maketrans('', '', string.punctuation)) #removing all ponctuations

    text = text.strip() # Remove leading/trailing whitespace
    text = re.sub("@[_A-Za-z0-9]+","",text) #Removing mention
    text = re.sub("[^\w\s#@/:%.,_-]", "", text, flags=re.UNICODE) #REmove emoji
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text) # Remove non-Arabic letters, symbols and numbers
    text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
    text = re.sub(r'\s*[A-Za-z]+\b', '' , text) #remove non-arabic word
    text = re.sub(r'#','',text) #removing hachtag
    text = re.sub(r'https?:\/\/\s+','',text)#remove the hyper link
    text = re.sub("\n","",text)
    text = re.sub(r'^[A-Za-z0-9.!?:؟]+'," ",text) ##Removing digits and punctuations
    text = re.sub("\n","",text)
    text = re.sub(u'\xa0','',text)
    text = re.sub(r'[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652]','',text)
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("[ااا]+","ا",text)
    text = re.sub("[يييي]+","ي",text)
    text = re.sub("[ةة]+","ة",text)
    text = re.sub("[ؤ]+","و",text)
    text = re.sub("[ججج]+","ج",text)
    text = re.sub("[ـــــــ]+","ـ",text)
    text = re.sub("[a-zA-Z]+","",text)
    text = re.sub("²", "", text)
    text = re.sub("[0-9]+","",text)
    text = re.sub("[ﷺöüçāīṣııšḥāḫםבםבḥāā]", "",text)
    text = re.sub("[헨리생일축하해요왕자어린손흥민]", "",text)
    text = re.sub(r'http', '',text)
    text = re.sub('[٠١٢٣٤٥٦٧٨٩]',"",text)
    text = re.sub('öü','',text)

    return text



  #============================= Stop Word Removal ============================

  def remove_stopwords(self, text):
    arabic_stopwords = set(stopwords.words('arabic'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if not word in arabic_stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text



  #=============================== Steamming ==============================

  def isHaraka(self, archar):
      """Checks for Arabic Harakat Marks (FATHA,DAMMA,KASRA,SUKUN,TANWIN).
      @param archar: arabic unicode char
      @type archar: unicode
      """
      if archar in HARAKAT:
          return True;
      else: return False;

  def isArabicword(self, word):
      """ Checks for an valid Arabic  word.
      An Arabic word not contains spaces, digits and pounctuation
      avoid some spelling error,  TEH_MARBUTA must be at the end.
      @param word: input word
      @type word: unicode
      @return: True if all charaters are in Arabic block
      @rtype: Boolean
      """
      if len(word)==0 : return False;
      elif re.search(u"([^\u0600-\u0652%s%s%s])"%(LAM_ALEF, LAM_ALEF_HAMZA_ABOVE,LAM_ALEF_MADDA_ABOVE),word):
          return False;
      elif self.isHaraka(word[0]) or word[0] in (WAW_HAMZA,YEH_HAMZA):
          return False;
  #  if Teh Marbuta or Alef_Maksura not in the end
      elif re.match(u"^(.)*[%s](.)+$"%ALEF_MAKSURA,word):
          return False;
      elif re.match(u"^(.)*[%s]([^%s%s%s])(.)+$"%(TEH_MARBUTA,DAMMA,KASRA,FATHA),word):
          return False;
      else:
          return True;


  def stem_token(self, token):
      ''' The method takes a valid Arabic word as a parameter and return a stemmed term '''
      if not self.isArabicword(token):
          return token

      if len(token) >= 6:
          prefixes = ['كال', 'بال', 'فال', 'مال', 'وال', 'ولل', 'است', 'يست', 'تست', 'مست']
          for prefix in prefixes:
              if token.startswith(prefix):
                  token = token[len(prefix):]
                  break

      # Remove prefixes/suffixes if the word length is greater than or equal to 5
      if len(token) >= 5:
          prefixes_suffixes = ['سن', 'سي', 'ست', 'لي', 'لن', 'لت', 'لل', 'ون', 'ات', 'ان', 'ين', 'تن', 'تم', 'كن', 'كم', 'هن', 'هم', 'يا', 'ني', 'تي', 'وا', 'ما', 'نا', 'ية', 'ها', 'ء']
          for ps in prefixes_suffixes:
              if token.startswith(ps):
                  token = token[len(ps):]
                  break
              elif token.endswith(ps):
                  token = token[:-len(ps)]
                  break

      if len(token) >= 4:
          prefixes_suffixes = ['ت', 'ي', 'ب', 'ل', 'ت', 'ة', 'ه', 'ا', 'ي']
          for ps in prefixes_suffixes:
              if token.startswith(ps):
                  token = token[len(ps):]
                  break
              elif token.endswith(ps):
                  token = token[:-len(ps)]
                  break

      if len(token) > 3 and token[:1] == '\u0648':
          token = token[1:]
      length = 0
      wordlen = len(token)

      for article in larkey_defarticles:
          length = len(article)
          if (wordlen > length + 1) and (token[:length] == article):
              token = token[length:]
              break

      if len(token) > 2:
          wordlen = len(token)
          for suffix in larkey_suffixes:
              suflen = len(suffix)
              if (wordlen > len(suffix) + 1) and token.endswith(suffix):
                  token = token[:wordlen - suflen]
                  wordlen = len(token)

      return token

  def arabic_preprocessing(self, text):
    # Remove 'ال'
    if len(text) > 3:
        text = re.sub(r'^ال', '', text)

    # Check for prefixes and remove them
    if len(text) >= 6:
        prefixes = ['كال', 'بال', 'فال', 'مال', 'وال', 'ولل', 'است', 'يست', 'تست', 'مست']
        suffixes = ['تان', 'تين', 'ان', 'ين', 'تن', 'تم', 'كم', 'كما']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        for suffixe in suffixes:
            if text.endswith(suffixe):
                text = text[:-len(suffixe)]
                break


    # Remove prefixes/suffixes if the word length is greater than or equal to 5
    if len(text) >= 5:
        prefixes_suffixes = ['سن', 'سي', 'ست', 'لي', 'لن', 'لت', 'لل', 'ون', 'ات',
                            'ين', 'تن', 'تم', 'كن', 'كم', 'هن', 'هم', 'يا',
                             'ني', 'تي', 'وا', 'ما', 'نا', 'ية', 'ها', 'ء', 'ل', 'تان']
        for ps in prefixes_suffixes:
            if text.startswith(ps):
                text = text[len(ps):]
                break
            elif text.endswith(ps):
                text = text[:-len(ps)]
                break

    # Remove prefixes/suffixes if the word length is greater than or equal to 4
    if len(text) >= 4:
        prefixes = ['ف']
        suffixes = ['ه', 'ة', 'ها', 'تا', 'هم', 'هن', 'ها', 'يا', 'كم', 'ها', 'نا']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        for suffixe in suffixes:
            if text.endswith(suffixe):
                text = text[:-len(suffixe)]
                break

    return text


  #==============================================(( MAIN ))==============================================

  def main(self, dataset):
    """
    Main function to preprocess the given dataset based on selected options.

    Parameters:
        self (object): The current instance of the class (assuming this function is inside a class).
        dataset (pandas DataFrame): The input dataset to be preprocessed.

    """

    # Record the start time for measuring the execution time.
    start = time.time()

    if self.column_stay == True:
        # Drop rows with any missing values in the dataset.
        dataset = dataset.dropna(axis=0, how='any')

        # Find the column with the maximum number of tokens in each row.
        max_col = dataset.columns[dataset.apply(lambda x: len(str(x).split())).argmax()]

        # Move the column with the most tokens to the first position in the DataFrame.
        cols = dataset.columns.tolist()
        cols.insert(0, cols.pop(cols.index(max_col)))
        dataset = dataset[cols]

        # Convert the data type of the first column to 'string'.
        column_name = dataset.columns[0]
        dataset = dataset.astype(dtype={column_name: 'string'})

    if self.column_stay == False:
        # Drop rows with any missing values in the dataset.
        dataset = dataset.dropna(axis=0, how='any')

        # If the dataset has more than one column, inform the user and automatically select the column
        # with the most tokens as the 'text' column.
        if len(dataset.columns) > 1:
            print('\033[1m' + '<WARNING>' + '\033[0m', ' Your dataset has a ', len(dataset.columns),
                  'columns. We will select the text column automatically according to the number of tokens in each column. \n If you want, you can terminate the process and drop the columns you dont want. \n')

        # Find the column with the maximum number of tokens in each row.
        max_col = dataset.columns[dataset.apply(lambda x: len(str(x).split())).argmax()]

        # Keep only the column with the most tokens as the 'text' column and rename it to 'text'.
        dataset = dataset[[max_col]]
        dataset = dataset.rename(columns={dataset.columns[0]: 'text'})

        # Convert the data type of the 'text' column to 'string'.
        column_name = dataset.columns[0]
        dataset = dataset.astype(dtype={column_name: 'string'})


    # Check if 'details' option is enabled.
    if self.details == True:

      # Print a message indicating the start of calculating original data statistics.
      print('\033[1m' + "Origninal data Stats calculating..." + '\033[0m' + "\n")

      # Calculate the number of tokens (words) in the original dataset before any pre-processing.
      dataset['tokens_count_original'] = dataset.iloc[:, 0].apply(self.get_token_count)
      total_tokens_original = dataset['tokens_count_original'].sum()
      print("Number of Tokens (words) in the orijinal dataset (before any pre-processing) :", total_tokens_original)

      # Calculate the number of characters in the original dataset before any pre-processing.
      dataset['char_count_original'] = dataset.iloc[:, 0].apply(self.get_char_count)
      total_chars_original = dataset['char_count_original'].sum()
      print("Number of Characters in the original dataset (before any pre-processing) :", total_chars_original, "\n")

      # Prepare dictionaries to store plot data for later visualization.
      plot_tokens = {'original dataset':total_tokens_original}
      plot_chars = {'original dataset':total_chars_original}

      print("-----------------------------\n\n")


      # If 'clean' option is specified in the pre-processing list.
      if 'clean' in self.process_list:

        # Print a message indicating the start of the data cleaning step.
        print('\033[1m' + "Data Cleaning step begain..." + '\033[0m' + "\n")

        # Drop rows with any missing values in the dataset.
        dataset = dataset.dropna(axis=0, how='any')

        # Apply the 'clean_texts' function to clean the text in the first column of the dataset.
        dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(lambda x: self.clean_texts(x))

        # Calculate the number of tokens (words) after the data cleaning step.
        dataset['tokens_count_after_cleaning'] = dataset.iloc[:, 0].apply(self.get_token_count)
        total_tokens_clean = dataset['tokens_count_after_cleaning'].sum()
        print("Number of Tokens (words) after (Data Cleaning) step :", total_tokens_clean)

        # Calculate the number of tokens (words) after the data cleaning step.
        dataset['char_count_after_cleaning'] = dataset.iloc[:, 0].apply(self.get_char_count)
        total_chars_clean = dataset['char_count_after_cleaning'].sum()
        print("Number of Characters after (Data Cleaning) step :", total_chars_clean, "\n")


        # If 'csv' option is enabled, save the cleaned dataset to a CSV file.
        if self.csv == True:

          dataset.to_csv('1after_Cleaning.csv',index=False)
          print("The CSV file of the dataset after data cleaning step has been created.")


        # If 'chart' option is enabled, update the plot data.
        if self.chart == True:
          plot_tokens['After data cleaning'] = total_tokens_clean
          plot_chars['After data cleaning'] = total_chars_clean

        print("-----------------------------\n\n")


      # If 'st_remove' option is specified in the pre-processing list.
      if 'st_remove' in self.process_list:

        # Print a message indicating the start of the stopword removal step.
        print('\033[1m' + "Stop Word Removal step begain..." + '\033[0m' + "\n")

        # Apply the 'remove_stopwords' function to remove stopwords from the text in the first column of the dataset.
        dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(self.remove_stopwords)

        # Remove custom stopwords defined in 'data_into_list'.
        arab_stopwords = r'\b(?:{})\b'.format('|'.join(data_into_list))
        dataset.iloc[:, 0] = dataset.iloc[:, 0].str.replace(arab_stopwords, '')

        # Remove additional custom stopwords defined in 'namess'.
        dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(lambda x: ' '.join([word for word in x.split() if word not in namess]))

        # Calculate the number of tokens (words) after the stopword removal step.
        dataset['tokens_count_after_SW_remove'] = dataset.iloc[:, 0].apply(self.get_token_count)
        total_tokens_SW = dataset['tokens_count_after_SW_remove'].sum()
        print("Number of Tokens (words) after (Data Cleaning) step :", total_tokens_SW)

        # Calculate the number of characters after the stopword removal step.
        dataset['char_count_after_SW_remove'] = dataset.iloc[:, 0].apply(self.get_char_count)
        total_chars_SW = dataset['char_count_after_SW_remove'].sum()
        print("Number of Characters after (Data Cleaning) step :", total_chars_SW, "\n")

        # If 'csv' option is enabled, save the dataset after stopword removal to a CSV file.
        if self.csv == True:
          dataset.to_csv('2after_Stop_word_remove.csv',index=False)
          print("The CSV file of the dataset after Stopwords removal has been created.")

        # If 'chart' option is enabled, update the plot data.
        if self.chart == True:
          plot_tokens['After remove stop words'] = total_tokens_SW
          plot_chars['After remove stop words'] = total_chars_SW

        print("-----------------------------\n\n")

      # If 'stemm' option is specified in the pre-processing list.
      if 'stemm' in self.process_list:

        # Print a message indicating the start of the text stemming step.
        print('\033[1m' + "Text Steamming step begain..." + '\033[0m' + "\n")

        # Perform stemming on the text in the dataset using 'stem_token' and 'arabic_preprocessing' functions.
        for o in range (len(dataset)):
          TEXT = dataset.iloc[:, 0].values[o]

          # Tokenize the text and perform stemming using 'stem_token'.
          split=TEXT.strip().split()
          result=[]
          for i in range(len(split)):
            stem = self.stem_token(split[i])
            result.append(stem)
          dataset.iloc[:, 0].values[o] = TreebankWordDetokenizer().detokenize(result)

        # Apply additional Arabic preprocessing using 'arabic_preprocessing'.
        for o in range (len(dataset)):
          TEXT = dataset.iloc[:, 0].values[o]

          split=TEXT.strip().split()
          result=[]
          for i in range(len(split)):
            stem = self.arabic_preprocessing(split[i])
            result.append(stem)
          dataset.iloc[:, 0].values[o] = TreebankWordDetokenizer().detokenize(result)

        # Calculate the number of tokens (words) after the text stemming step.
        dataset['tokens_count_after_ST'] = dataset.iloc[:, 0].apply(self.get_token_count)
        total_tokens_ST = dataset['tokens_count_after_ST'].sum()
        print("Number of Tokens (words) after (Text Steamming) step :", total_tokens_ST)

        # Calculate the number of characters after the text stemming step.
        dataset['char_count_after_ST'] = dataset.iloc[:, 0].apply(self.get_char_count)
        total_chars_ST = dataset['char_count_after_ST'].sum()
        print("Number of Characters after (Text Steamming) step :", total_chars_ST, "\n")

        # If 'csv' option is enabled, save the dataset after stemming to a CSV file.
        if self.csv == True:
          dataset.to_csv('3after_stemming.csv',index=False)
          print("The CSV file of the dataset after data stemming step has been created (Which contain all data pre-processing you want).")

        # If 'chart' option is enabled, update the plot data.
        if self.chart == True:
          plot_tokens['After data stemming'] = total_tokens_ST
          plot_chars['After data stemming'] = total_chars_ST

        print("-----------------------------\n")
        end = time.time()
        print('Time required to complete pre-processing steps:', round((end-start), 2), ' second ', '(', round((end-start)/60, 2), ' minute). \n')
        print("-----------------------------\n\n")


      # If 'chart' option is enabled, create and display a chart comparing the number of tokens and characters
      # after each pre-processing step.
      if self.chart == True:
        data = plot_tokens
        data2 = plot_chars

        X = np.arange(len(data))
        fig = plt.figure(figsize=(20,5))

        ax1 = fig.add_subplot(121)
        ax1.bar(X , data.values(), color = 'g', width = 0.5)
        plt.xticks(X,data.keys())
        ax1.set_title('Number of tokens after each pre-processing step')
        ax1.set_ylabel('Number of tokens')

        ax2 = fig.add_subplot(122)
        ax2.bar(X , data2.values(), color = 'b', width = 0.5)
        plt.xticks(X,data2.keys())
        ax2.set_title('Number of charecters after each pre-processing step')
        ax2.set_ylabel('Number of charecters')

        plt.show()
        plt.savefig('chart.png')

      # If 'chart' option is False, create and save the chart as 'chart.png' without displaying it.
      if self.chart == False:
        data = plot_tokens
        data2 = plot_chars

        X = np.arange(len(data))
        fig = plt.figure(figsize=(20,5))

        ax1 = fig.add_subplot(121)
        ax1.bar(X , data.values(), color = 'g', width = 0.5)
        plt.xticks(X,data.keys())
        ax1.set_title('Number of tokens after each pre-processing step')
        ax1.set_ylabel('Number of tokens')

        ax2 = fig.add_subplot(122)
        ax2.bar(X , data2.values(), color = 'b', width = 0.5)
        plt.xticks(X,data2.keys())
        ax2.set_title('Number of charecters after each pre-processing step')
        ax2.set_ylabel('Number of charecters')

        plt.savefig('chart.png')
        plt.close()


    else:

      # Calculate the number of tokens (words) in the original dataset before any pre-processing.
      dataset['tokens_count_original'] = dataset.iloc[:, 0].apply(self.get_token_count)
      total_tokens_original = dataset['tokens_count_original'].sum()

      # Calculate the number of characters in the original dataset before any pre-processing.
      dataset['char_count_original'] = dataset.iloc[:, 0].apply(self.get_char_count)
      total_chars_original = dataset['char_count_original'].sum()

      # Prepare dictionaries to store plot data for later visualization.
      plot_tokens = {'original dataset':total_tokens_original}
      plot_chars = {'original dataset':total_chars_original}


      # If 'clean' option is specified in the pre-processing list.
      if 'clean' in self.process_list:

        # Drop rows with any missing values in the dataset.
        dataset = dataset.dropna(axis=0, how='any')

        # Apply the 'clean_texts' function to clean the text in the first column of the dataset.
        dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(lambda x: self.clean_texts(x))

        # Calculate the number of tokens (words) after the data cleaning step.
        dataset['tokens_count_after_cleaning'] = dataset.iloc[:, 0].apply(self.get_token_count)
        total_tokens_clean = dataset['tokens_count_after_cleaning'].sum()

        # Calculate the number of tokens (words) after the data cleaning step.
        dataset['char_count_after_cleaning'] = dataset.iloc[:, 0].apply(self.get_char_count)
        total_chars_clean = dataset['char_count_after_cleaning'].sum()


        # If 'csv' option is enabled, save the cleaned dataset to a CSV file.
        if self.csv == True:
          dataset.to_csv('1after_Cleaning.csv',index=False)

        # If 'chart' option is enabled, update the plot data.
        if self.chart == True:
          plot_tokens['After data cleaning'] = total_tokens_clean
          plot_chars['After data cleaning'] = total_chars_clean


      # If 'st_remove' option is specified in the pre-processing list.
      if 'st_remove' in self.process_list:

        # Apply the 'remove_stopwords' function to remove stopwords from the text in the first column of the dataset.
        dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(self.remove_stopwords)

        # Remove custom stopwords defined in 'data_into_list'.
        arab_stopwords = r'\b(?:{})\b'.format('|'.join(data_into_list))
        dataset.iloc[:, 0]=dataset.iloc[:, 0].str.replace(arab_stopwords, '')

        # Remove additional custom stopwords defined in 'namess'.
        dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(lambda x: ' '.join([word for word in x.split() if word not in namess]))

        # Calculate the number of tokens (words) after the stopword removal step.
        dataset['tokens_count_after_SW_remove'] = dataset.iloc[:, 0].apply(self.get_token_count)
        total_tokens_SW = dataset['tokens_count_after_SW_remove'].sum()

        # Calculate the number of characters after the stopword removal step.
        dataset['char_count_after_SW_remove'] = dataset.iloc[:, 0].apply(self.get_char_count)
        total_chars_SW = dataset['char_count_after_SW_remove'].sum()

        # If 'csv' option is enabled, save the dataset after stopword removal to a CSV file.
        if self.csv == True:
          dataset.to_csv('2after_Stop_word_remove.csv',index=False)

        # If 'chart' option is enabled, update the plot data.
        if self.chart == True:
          plot_tokens['After remove stop words'] = total_tokens_SW
          plot_chars['After remove stop words'] = total_chars_SW


      # If 'stemm' option is specified in the pre-processing list.
      if 'stemm' in self.process_list:

        # Perform stemming on the text in the dataset using 'stem_token' and 'arabic_preprocessing' functions.
        for o in range (len(dataset)):
          TEXT = dataset.iloc[:, 0].values[o]
          split=TEXT.strip().split()
          result=[]
          for i in range(len(split)):
            stem = self.stem_token(split[i])
            result.append(stem)
          dataset.iloc[:, 0].values[o] = TreebankWordDetokenizer().detokenize(result)

        for o in range (len(dataset)):
          TEXT = dataset.iloc[:, 0].values[o]
          split=TEXT.strip().split()
          result=[]
          for i in range(len(split)):
            stem = self.arabic_preprocessing(split[i])
            result.append(stem)
          dataset.iloc[:, 0].values[o] = TreebankWordDetokenizer().detokenize(result)

        # Calculate the number of tokens (words) after the text stemming step.
        dataset['tokens_count_after_ST'] = dataset.iloc[:, 0].apply(self.get_token_count)
        total_tokens_ST = dataset['tokens_count_after_ST'].sum()

        # Calculate the number of characters after the text stemming step.
        dataset['char_count_after_ST'] = dataset.iloc[:, 0].apply(self.get_char_count)
        total_chars_ST = dataset['char_count_after_ST'].sum()

        # If 'csv' option is enabled, save the dataset after text stemming to a CSV file.
        if self.csv == True:
          dataset.to_csv('3after_stemming.csv',index=False)

        # If 'chart' option is enabled, update the plot data.
        if self.chart == True:
          plot_tokens['After data stemming'] = total_tokens_ST
          plot_chars['After data stemming'] = total_chars_ST


      # If 'chart' option is enabled, create and save the chart comparing the number of tokens and characters
      # after each pre-processing step.
      if self.chart == True:
        data = plot_tokens
        data2 = plot_chars

        X = np.arange(len(data))
        fig = plt.figure(figsize=(20,5))

        ax1 = fig.add_subplot(121)
        ax1.bar(X , data.values(), color = 'g', width = 0.5)
        plt.xticks(X,data.keys())
        ax1.set_title('Number of tokens after each pre-processing step')
        ax1.set_ylabel('Number of tokens')

        ax2 = fig.add_subplot(122)
        ax2.bar(X , data2.values(), color = 'b', width = 0.5)
        plt.xticks(X,data2.keys())
        ax2.set_title('Number of charecters after each pre-processing step')
        ax2.set_ylabel('Number of charecters')

        plt.savefig('chart.png')

      # If 'chart' option is False, create and save the chart as 'chart.png' without displaying it.
      if self.chart == False:
        data = plot_tokens
        data2 = plot_chars

        X = np.arange(len(data))
        fig = plt.figure(figsize=(20,5))

        ax1 = fig.add_subplot(121)
        ax1.bar(X , data.values(), color = 'g', width = 0.5)
        plt.xticks(X,data.keys())
        ax1.set_title('Number of tokens after each pre-processing step')
        ax1.set_ylabel('Number of tokens')

        ax2 = fig.add_subplot(122)
        ax2.bar(X , data2.values(), color = 'b', width = 0.5)
        plt.xticks(X,data2.keys())
        ax2.set_title('Number of charecters after each pre-processing step')
        ax2.set_ylabel('Number of charecters')

        plt.savefig('chart.png')
        plt.close()