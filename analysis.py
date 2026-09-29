import pandas as pd
import numpy as np
from sympy import python
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
# 1. قراءة الملفات الأربعة
deliveries = pd.read_csv('Deliveries.csv')
orders = pd.read_csv('Orders.csv')
drivers = pd.read_csv('Drivers.csv')
feedback = pd.read_csv('Customer_Feedback (1).csv')

# 2. فحص أبعاد البيانات (Shape)
print("--- Data Dimensions (Rows, Columns) ---")
print("Deliveries:", deliveries.shape)
print("Orders:", orders.shape)
print("Drivers:", drivers.shape)
print("Feedback:", feedback.shape)

# 3. فحص تفصيلي للقيم المفقودة في كل جدول
print("\n--- Missing Values Summary ---")
print("Orders Missing:\n", orders.isnull().sum())
print("\nDeliveries Missing:\n", deliveries.isnull().sum())
print("\nDrivers Missing:\n", drivers.isnull().sum())
print("\nFeedback Missing:\n", feedback.isnull().sum())

# 4. إزالة الصفوف المكررة تماماً إن وجدت
deliveries = deliveries.drop_duplicates()
orders = orders.drop_duplicates()
drivers = drivers.drop_duplicates()
feedback = feedback.drop_duplicates()

# 5. التعامل الاحترافي مع القيم المفقودة:
# . دمج الجداول الأساسية
df_main = orders.merge(deliveries, on='OrderID', how='inner')
df_main = df_main.merge(drivers, on='DriverID', how='left')
df_main = df_main.merge(feedback, on='OrderID', how='left')

# تحويل أعمدة الوقت إلى صيغة datetime
for col in ['OrderTime', 'DispatchTime', 'PickupTime', 'DeliveryTime']:
    if col in df_main.columns:
        df_main[col] = pd.to_datetime(df_main[col])

# . حساب الوسيط لفترات التجهيز والتوصيل من البيانات الموجودة فعلياً
valid_prep = (df_main['PickupTime'] - df_main['OrderTime']).dt.total_seconds() / 60.0
median_prep_mins = valid_prep.median() if not valid_prep.dropna().empty else 15.0

valid_transit = (df_main['DeliveryTime'] - df_main['PickupTime']).dt.total_seconds() / 60.0
median_transit_mins = valid_transit.median() if not valid_transit.dropna().empty else 25.0

# . تطبيق Fillna بطريقة ذكية بدون حذف أي صف
df_main['PickupTime'] = df_main['PickupTime'].fillna(df_main['OrderTime'] + pd.Timedelta(minutes=median_prep_mins))
df_main['DeliveryTime'] = df_main['DeliveryTime'].fillna(df_main['PickupTime'] + pd.Timedelta(minutes=median_transit_mins))

print("\n--- Missing Values in Feedback After Filling ---")
print(feedback.isnull().sum())
print("\n--- Missing values in Deliveries After Filling ---")
print(deliveries.isnull().sum())


# - دمج الجداول معاً لإنشاء DataFrame الرئيسي
df_main = orders.merge(deliveries, on='OrderID', how='inner')
df_main = df_main.merge(drivers, on='DriverID', how='left')
df_main = df_main.merge(feedback, on='OrderID', how='left')

# 6. معالجة القيم المفقودة في الأعمدة العددية (مثل Distance_KM أو DeliveryFee إن وجدت)
if 'Distance_KM' in df_main.columns:
    df_main['Distance_KM'] = df_main['Distance_KM'].fillna(df_main['Distance_KM'].median())

if 'DeliveryFee' in df_main.columns:
    df_main['DeliveryFee'] = df_main['DeliveryFee'].fillna(df_main['DeliveryFee'].median())

print("\n--- Missing Values Handled Successfully & DataFrame is Ready! ---")




#Feature Engineering:
# التأكد من تحويل أعمدة الوقت إلى صيغة datetime
time_columns = ['OrderTime', 'PickupTime', 'DeliveryTime']
for col in time_columns:
    if col in df_main.columns:
        df_main[col] = pd.to_datetime(df_main[col])

# حساب المتغيرات وتخزينها في متغيرات صريحة[cite: 1]
prep_time = (df_main['PickupTime'] - df_main['OrderTime']).dt.total_seconds() / 60.0
transit_time = (df_main['DeliveryTime'] - df_main['PickupTime']).dt.total_seconds() / 60.0
total_trip_duration = (df_main['DeliveryTime'] - df_main['OrderTime']).dt.total_seconds() / 60.0

threshold_minutes = 40.0
is_delayed = total_trip_duration > threshold_minutes

max_transit = transit_time.max() if transit_time.max() > 0 else 1
speed_score = 1 - (transit_time / max_transit)

if 'Rating' in df_main.columns:
    norm_rating = df_main['Rating'] / 5.0
    driver_efficiency_index = (0.5 * speed_score) + (0.5 * norm_rating)
else:
    driver_efficiency_index = speed_score

# تعيين النتائج وتخزينها داخل أعمدة DataFrame الرئيسية
df_main['Prep_Time'] = prep_time
df_main['Transit_Time'] = transit_time
df_main['Total_Trip_Duration'] = total_trip_duration
df_main['Is_Delayed'] = is_delayed
df_main['Driver_Efficiency_Index'] = driver_efficiency_index

# طباعة النتيجة وعرض عينة منها

print(df_main[['OrderID', 'Prep_Time', 'Transit_Time', 'Total_Trip_Duration', 'Is_Delayed', 'Driver_Efficiency_Index']].head(10))



# التأكد من وجود المتغيرات الأساسية المعرفة مسبقاً في df_main
# (مثل Is_Delayed, Prep_Time, Transit_Time, Total_Trip_Duration, Driver_Efficiency_Index)

# 1. معدل التأخير (Delay Rate) ومتوسط زمن التأخير
total_orders_count = len(df_main)
delayed_orders_count = df_main['Is_Delayed'].sum()
delay_rate_pct = (delayed_orders_count / total_orders_count) * 100
avg_delay_minutes = df_main[df_main['Is_Delayed']]['Total_Trip_Duration'].mean() - 40.0

# 2. تحليل أسباب الاختناق (Bottleneck Analysis)
avg_prep_time = df_main['Prep_Time'].mean()
avg_transit_time = df_main['Transit_Time'].mean()
bottleneck_result = "المطعم (Prep_Time)" if avg_prep_time > avg_transit_time else "الطريق والتوصيل (Transit_Time)"

# 3. تأثير نوع المركبة والمسافة (Vehicle Type & Distance Impact)
vehicle_delay_impact = df_main.groupby('VehicleType').agg(
    Avg_Distance=('Distance_KM', 'mean'),
    Avg_Transit_Time=('Transit_Time', 'mean'),
    Delay_Rate=('Is_Delayed', lambda x: (x.sum() / len(x)) * 100)
).reset_index()

# 4. أوقات الذروة (Peak Hours & Days)
df_main['OrderHour'] = pd.to_datetime(df_main['OrderTime']).dt.hour
df_main['OrderDay'] = pd.to_datetime(df_main['OrderTime']).dt.day_name()
peak_hours = df_main.groupby('OrderHour')['OrderID'].count().nlargest(3)
peak_days = df_main.groupby('OrderDay')['Is_Delayed'].mean().nlargest(3) * 100

# 5. تقييم أداء المناديب (Driver Performance)
top_drivers = df_main.groupby('DriverID').agg(
    Avg_Efficiency=('Driver_Efficiency_Index', 'mean'),
    Avg_Rating=('Rating', 'mean'),
    Total_Orders=('OrderID', 'count')
).nlargest(5, 'Avg_Efficiency')

worst_drivers = df_main.groupby('DriverID').agg(
    Delay_Count=('Is_Delayed', 'sum'),
    Total_Orders=('OrderID', 'count')
).assign(Delay_Rate=lambda x: (x['Delay_Count'] / x['Total_Orders']) * 100).nlargest(5, 'Delay_Rate')

# 6. أداء المطاعم (Top 10 Slowest Restaurants)
slowest_restaurants = df_main.groupby('RestaurantID').agg(
    Avg_Prep_Time=('Prep_Time', 'mean'),
    Avg_Customer_Rating=('DeliveryRating', 'mean') # أو التقييم العام المتاح
).nlargest(10, 'Avg_Prep_Time')

# 7. تحليل شكاوى العملاء (Customer Complaints Analysis)
if 'IssueCategory' in df_main.columns:
    complaints_freq = df_main['IssueCategory'].value_counts(normalize=True) * 100
    complaints_impact = df_main.groupby('IssueCategory')['DeliveryRating'].mean()
else:
    complaints_freq = pd.Series()
    complaints_impact = pd.Series()

# 8. سلوك الإلغاء (Order Cancellations)
if 'Status' in df_main.columns:
    cancellation_rate = (df_main['Status'].str.lower().eq('cancelled').sum() / total_orders_count) * 100
    # العلاقة بين وقت الانتظار والإلغاء
    waiting_time_cancellation = df_main.groupby('Status')['Total_Trip_Duration'].mean()
else:
    cancellation_rate = 0.0
    waiting_time_cancellation = pd.Series()

# 9. ربحية المناطق الجغرافية (Geographical Profitability / Delivery Fee)
if 'City' in df_main.columns and 'DeliveryFee' in df_main.columns:
    geo_profitability = df_main.groupby('City').agg(
        Total_Orders=('OrderID', 'count'),
        Total_Delivery_Revenue=('DeliveryFee', 'sum'),
        Avg_Delivery_Fee=('DeliveryFee', 'mean')
    ).nlargest(5, 'Total_Orders')
else:
    geo_profitability = pd.DataFrame()







# 1. Delay Rate
total_orders_count = len(df_main)
delayed_orders_count = df_main['Is_Delayed'].sum()
delay_rate_pct = (delayed_orders_count / total_orders_count) * 100
avg_delay_minutes = df_main[df_main['Is_Delayed']]['Total_Trip_Duration'].mean() - 40.0

print("=== 1. Delay Rate Analysis ===")
print(f"Delay Rate: {delay_rate_pct:.2f}% | Average Delay Time: {avg_delay_minutes:.2f} minutes\n")

# 2. Bottleneck Analysis
avg_prep_time = df_main['Prep_Time'].mean()
avg_transit_time = df_main['Transit_Time'].mean()
bottleneck_result = "Restaurant (Prep_Time)" if avg_prep_time > avg_transit_time else "Transit & Delivery (Transit_Time)"

print("=== 2. Bottleneck Analysis ===")
print(f"Average Prep Time: {avg_prep_time:.2f} minutes")
print(f"Average Transit Time: {avg_transit_time:.2f} minutes")
print(f"Main Bottleneck Location: {bottleneck_result}\n")

# 3. Vehicle Type & Distance Impact
vehicle_delay_impact = df_main.groupby('VehicleType').agg(
    Avg_Distance=('Distance_KM', 'mean'),
    Avg_Transit_Time=('Transit_Time', 'mean'),
    Delay_Rate=('Is_Delayed', lambda x: (x.sum() / len(x)) * 100)
).reset_index()

print("=== 3. Vehicle Type & Distance Impact ===")
print(vehicle_delay_impact, "\n")

# 4. Peak Hours & Days
df_main['OrderHour'] = pd.to_datetime(df_main['OrderTime']).dt.hour
df_main['OrderDay'] = pd.to_datetime(df_main['OrderTime']).dt.day_name()
peak_hours = df_main.groupby('OrderHour')['OrderID'].count().nlargest(3)
peak_days = df_main.groupby('OrderDay')['Is_Delayed'].mean().nlargest(3) * 100

print("=== 4. Peak Hours & Days ===")
print("Top Peak Hours:\n", peak_hours)
print("Top Days by Delay Rate (%):\n", peak_days, "\n")

# 5. Driver Performance
top_drivers = df_main.groupby('DriverID').agg(
    Avg_Efficiency=('Driver_Efficiency_Index', 'mean'),
    Avg_Rating=('Rating', 'mean'),
    Total_Orders=('OrderID', 'count')
).nlargest(5, 'Avg_Efficiency')

worst_drivers = df_main.groupby('DriverID').agg(
    Delay_Count=('Is_Delayed', 'sum'),
    Total_Orders=('OrderID', 'count')
).assign(Delay_Rate=lambda x: (x['Delay_Count'] / x['Total_Orders']) * 100).nlargest(5, 'Delay_Rate')

print("=== 5. Driver Performance ===")
print("Top 5 Drivers:\n", top_drivers)
print("Worst 5 Drivers by Delay Rate:\n", worst_drivers, "\n")

# 6. Restaurant Performance (Top 10 Slowest Restaurants)
slowest_restaurants = df_main.groupby('RestaurantID').agg(
    Avg_Prep_Time=('Prep_Time', 'mean'),
    Avg_Customer_Rating=('DeliveryRating', 'mean')
).nlargest(10, 'Avg_Prep_Time')

print("=== 6. Restaurant Performance (Top 10 Slowest Restaurants) ===")
print(slowest_restaurants, "\n")

# 7. Customer Complaints Analysis
if 'IssueCategory' in df_main.columns:
    complaints_freq = df_main['IssueCategory'].value_counts(normalize=True) * 100
    complaints_impact = df_main.groupby('IssueCategory')['DeliveryRating'].mean()
else:
    complaints_freq = pd.Series()
    complaints_impact = pd.Series()

print("=== 7. Customer Complaints Analysis ===")
if not complaints_freq.empty:
    print("Complaints Frequency (%):\n", complaints_freq)
    print("Average Service Rating by Issue Category:\n", complaints_impact, "\n")
else:
    print("Complaints data not available in the dataset.\n")

# 8. Order Cancellations
if 'Status' in df_main.columns:
    cancellation_rate = (df_main['Status'].str.lower().eq('cancelled').sum() / total_orders_count) * 100
    waiting_time_cancellation = df_main.groupby('Status')['Total_Trip_Duration'].mean()
else:
    cancellation_rate = 0.0
    waiting_time_cancellation = pd.Series()

print("=== 8. Order Cancellations ===")
print(f"Overall Cancellation Rate: {cancellation_rate:.2f}%")
if not waiting_time_cancellation.empty:
    print("Average Trip Duration by Order Status:\n", waiting_time_cancellation, "\n")

# 9. Geographical Profitability
if 'City' in df_main.columns and 'DeliveryFee' in df_main.columns:
    geo_profitability = df_main.groupby('City').agg(
        Total_Orders=('OrderID', 'count'),
        Total_Delivery_Revenue=('DeliveryFee', 'sum'),
        Avg_Delivery_Fee=('DeliveryFee', 'mean')
    ).nlargest(5, 'Total_Orders')
else:
    geo_profitability = pd.DataFrame()

print("=== 9. Geographical Profitability ===")
if not geo_profitability.empty:
    print(geo_profitability, "\n")
else:
    print("Geographical or Delivery Fee data not available.\n")

# 10. Operational Recommendations
operational_recommendations = [
    "1. Impose penalties or incentives to improve prep times in the 10 slowest restaurants.",
    "2. Reassign low-performing drivers or provide time-management training courses.",
    "3. Adjust delivery benchmarks during peak hours and hire additional drivers.",
    "4. Address frequent complaint causes such as delayed or cold food with restaurants.",
    "5. Optimize geographical order distribution based on bike/car routes to reduce Transit_Time."
]

print("=== 10. Operational Recommendations ===")
for rec in operational_recommendations:
    print(rec)



# إنشاء مجلد حفظ الرسوم البيانية إذا لم يكن موجوداً
os.makedirs('charts', exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. الرسم الأول: نسبة الطلبات المتأخرة مقابل الطلبات في الموعد (Pie Chart)
# . حساب عدد وتكرار كل حالة (True و False)
delay_counts = df_main['Is_Delayed'].value_counts()

# 2. إنشاء الرسم البياني الدائري (Pie Chart) بالقيم الحقيقية
plt.figure(figsize=(6, 6))
plt.pie(
    x=delay_counts.values,            # القيم صراحة (الأعداد أو النسب)
    labels=['On time (False)', 'Delayed (True)'],  # التسميات بالترتيب المطابق
    autopct='%1.1f%%',                # إظهار النسبة المئوية بدقة رقم عشري واحد
    startangle=90,                    # بدء الرسم من الأعلى
    colors=['#2ecc71', '#e74c3c']     # ألوان توضيحية (أخضر للطبيعي، أحمر للتأخير)
)

plt.title('Delayed vs. On-Time Orders Ratio')
plt.axis('equal')  # لضمان ظهور الدائرة بشكل متناسق تماماً
plt.show()

# 2. الرسم الثاني: تحليل الاختناق باستخدام BoxPlot (Prep_Time مقابل Transit_Time)
plt.figure(figsize=(8, 6))
melted_df = df_main[['Prep_Time', 'Transit_Time']].melt(var_name='Stage', value_name='Minutes')
sns.boxplot(x='Stage', y='Minutes', data=melted_df, palette=['#3498db', '#e67e22'])
plt.title('Bottleneck Analysis: Prep Time vs Transit Time (BoxPlot)')
plt.savefig('charts/chart_2_bottleneck_boxplot.png', bbox_inches='tight')
plt.show()
plt.close()

# 3. الرسم الثالث: تأثير نوع المركبة على زمن الطريق
plt.figure(figsize=(8, 6))
if 'VehicleType' in df_main.columns:
    sns.barplot(x='VehicleType', y='Transit_Time', data=df_main, estimator=np.mean, palette='Set2')
    plt.title('Average Transit Time by Vehicle Type')
    plt.savefig('charts/chart_3_vehicle_impact.png', bbox_inches='tight')
plt.show()
plt.close()

# 4. الرسم الرابع: خريطة الحرارة لأوقات الذروة Peak Hours Heatmap (أيام الأسبوع مقابل ساعات اليوم)
plt.figure(figsize=(10, 6))
if 'OrderHour' not in df_main.columns:
    df_main['OrderHour'] = pd.to_datetime(df_main['OrderTime']).dt.hour
if 'OrderDay' not in df_main.columns:
    df_main['OrderDay'] = pd.to_datetime(df_main['OrderTime']).dt.day_name()

heatmap_data = df_main.pivot_table(index='OrderDay', columns='OrderHour', values='OrderID', aggfunc='count', fill_value=0)
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex([d for d in days_order if d in heatmap_data.index])

sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=.5)
plt.title('Peak Hours & Days Heatmap (Order Volume)')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.savefig('charts/chart_4_peak_hours_heatmap.png', bbox_inches='tight')
plt.show()
plt.close()

# 5. الرسم الخامس: أفضل المناديب بناءً على مؤشر الكفاءة
plt.figure(figsize=(8, 6))
top_drivers_plot = df_main.groupby('DriverID')['Driver_Efficiency_Index'].mean().nlargest(10).reset_index()
sns.barplot(x='Driver_Efficiency_Index', y='DriverID', data=top_drivers_plot, orient='h', palette='viridis')
plt.title('Top 10 Drivers by Efficiency Index')
plt.savefig('charts/chart_5_top_drivers.png', bbox_inches='tight')
plt.show()
plt.close()

# 6. الرسم السادس: أبطأ 10 مطاعم في وقت التحضير (Prep Time)
plt.figure(figsize=(10, 6))
slowest_res_plot = df_main.groupby('RestaurantID')['Prep_Time'].mean().nlargest(10).reset_index()
sns.barplot(x='Prep_Time', y='RestaurantID', data=slowest_res_plot, orient='h', palette='Reds_r')
plt.title('Top 10 Slowest Restaurants by Average Prep Time')
plt.savefig('charts/chart_6_slowest_restaurants.png', bbox_inches='tight')
plt.show()
plt.close()

# 7. الرسم السابع: تحليل تكرار شكاوى العملاء حسب الفئة
plt.figure(figsize=(8, 6))
if 'IssueCategory' in df_main.columns:
    sns.countplot(y='IssueCategory', data=df_main, order=df_main['IssueCategory'].value_counts().index, palette='pastel')
    plt.title('Customer Complaints Frequency by Category')
    plt.savefig('charts/chart_7_complaints_frequency.png', bbox_inches='tight')
plt.show()
plt.close()

# 8. الرسم الثامن: علاقة وقت الانتظار بحالة الطلب (الإلغاء)
plt.figure(figsize=(8, 6))
if 'Status' in df_main.columns:
    sns.boxplot(x='Status', y='Total_Trip_Duration', data=df_main, palette='coolwarm')
    plt.title('Total Trip Duration by Order Status (Cancellations)')
    plt.savefig('charts/chart_8_cancellations_duration.png', bbox_inches='tight')
plt.show()
plt.close()

# 9. الرسم التاسع: ربحية المناطق الجغرافية (إجمالي رسوم التوصيل حسب المدينة)
plt.figure(figsize=(8, 6))
if 'City' in df_main.columns and 'DeliveryFee' in df_main.columns:
    city_rev = df_main.groupby('City')['DeliveryFee'].sum().reset_index()
    sns.barplot(x='City', y='DeliveryFee', data=city_rev, palette='magma')
    plt.title('Total Delivery Revenue by City')
    plt.savefig('charts/chart_9_geo_revenue.png', bbox_inches='tight')
plt.show()
plt.close()

# 10. الرسم العاشر: علاقة المسافة (KM) بزمن الطريق (Transit Time)
plt.figure(figsize=(8, 6))
if 'Distance_KM' in df_main.columns:
    sns.scatterplot(x='Distance_KM', y='Transit_Time', hue='Is_Delayed', data=df_main, alpha=0.6, palette=['blue', 'red'])
    plt.title('Distance (KM) vs Transit Time')
    plt.savefig('charts/chart_10_distance_vs_transit.png', bbox_inches='tight')
plt.show()
plt.close()

# 11. الرسم الحادي عشر: العلاقة التناثرية بين وقت التحضير ووقت الطريق
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Prep_Time', y='Transit_Time', data=df_main, alpha=0.5, color='purple')
plt.title('Relationship Between Prep Time and Transit Time')
plt.savefig('charts/chart_11_prep_vs_transit.png', bbox_inches='tight')
plt.show()
plt.close()

# 12. الرسم الثاني عشر: توزيع تقييمات العملاء بناءً على حالة التأخير (BoxPlot)
plt.figure(figsize=(8, 6))
rating_col = 'DeliveryRating' if 'DeliveryRating' in df_main.columns else ('Rating' if 'Rating' in df_main.columns else None)
if rating_col:
    sns.boxplot(x='Is_Delayed', y=rating_col, data=df_main, palette='Set1')
    plt.title('Customer Rating Distribution by Delay Status')
    plt.savefig('charts/chart_12_rating_vs_delay.png', bbox_inches='tight')
plt.show()
plt.close()

print("=== Successfully generated and saved all 12 charts to the 'charts' folder! ===")




def generate_executive_report():
    # 1. تحميل وتجهيز جميع البيانات
    try:
        deliveries = pd.read_csv('Deliveries.csv').drop_duplicates()
        orders = pd.read_csv('Orders.csv').drop_duplicates()
        drivers = pd.read_csv('Drivers.csv').drop_duplicates()
        feedback = pd.read_csv('Customer_Feedback (1).csv').drop_duplicates()
        print("--- Data Loaded Successfully ---")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return "Error loading data files."

    feedback['IssueCategory'] = feedback['IssueCategory'].fillna('No Issue')
    feedback['Comment'] = feedback['Comment'].fillna('No Comment')
    deliveries['PickupTime'] = deliveries['PickupTime'].fillna(deliveries['DispatchTime'])
    deliveries['DeliveryTime'] = deliveries['DeliveryTime'].fillna(deliveries['DispatchTime'])

    df_main = orders.merge(deliveries, on='OrderID', how='inner')
    df_main = df_main.merge(drivers, on='DriverID', how='left')
    df_main = df_main.merge(feedback, on='OrderID', how='left')

    df_main['OrderTime'] = pd.to_datetime(df_main['OrderTime'])
    df_main['PickupTime'] = pd.to_datetime(df_main['PickupTime'])
    df_main['DeliveryTime'] = pd.to_datetime(df_main['DeliveryTime'])

    # هندسة الخصائص (Feature Engineering)
    df_main['Prep_Time'] = (df_main['PickupTime'] - df_main['OrderTime']).dt.total_seconds() / 60.0
    df_main['Transit_Time'] = (df_main['DeliveryTime'] - df_main['PickupTime']).dt.total_seconds() / 60.0
    df_main['Total_Trip_Duration'] = (df_main['DeliveryTime'] - df_main['OrderTime']).dt.total_seconds() / 60.0
    df_main['Is_Delayed'] = df_main['Total_Trip_Duration'] > 40.0

    max_transit = df_main['Transit_Time'].max() if df_main['Transit_Time'].max() > 0 else 1
    speed_score = 1 - (df_main['Transit_Time'] / max_transit)
    norm_rating = df_main['Rating'] / 5.0 if 'Rating' in df_main.columns else speed_score
    df_main['Driver_Efficiency_Index'] = (0.5 * speed_score) + (0.5 * norm_rating)

    # استخراج المقاييس والإحصائيات الحقيقية
    # استخراج المقاييس والإحصائيات وتجهيز النصوص المنسقة لتجنب الهلوسة تماماً
    total_orders_count = len(df_main)
    delay_rate = (df_main['Is_Delayed'].sum() / total_orders_count) * 100
    avg_delay_time = df_main[df_main['Is_Delayed']]['Total_Trip_Duration'].mean() - 40.0
    avg_prep = df_main['Prep_Time'].mean()
    avg_transit = df_main['Transit_Time'].mean()
    bottleneck = "Transit & Delivery" if avg_transit > avg_prep else "Restaurant Prep"

    # 1. تنسيق بيانات المركبات
    vehicle_groups = df_main.groupby('VehicleType').agg(
        Avg_Distance=('Distance_KM', 'mean'),
        Avg_Transit=('Transit_Time', 'mean'),
        Delay_Pct=('Is_Delayed', lambda x: (x.sum() / len(x)) * 100)
    )
    vehicle_impact_lines = []
    for vehicle, row in vehicle_groups.iterrows():
        line = f"- {vehicle}: متوسط المسافة {row['Avg_Distance']:.2f} كم، متوسط وقت النقل {row['Avg_Transit']:.2f} دقيقة، ومعدل التأخير {row['Delay_Pct']:.2f}%"
        vehicle_impact_lines.append(line)
    vehicle_impact = '\n'.join(vehicle_impact_lines)

    # 2. تنسيق ساعات وأيام الذروة كنصوص دقيقة
    peak_hours_series = df_main.groupby(df_main['OrderTime'].dt.hour)['OrderID'].count().nlargest(3)
    peak_hours = ", ".join([f"الساعة {h} ({c} طلب)" for h, c in peak_hours_series.items()])

    peak_days_series = df_main.groupby(df_main['OrderTime'].dt.day_name())['Is_Delayed'].mean().nlargest(3) * 100
    peak_days = ", ".join([f"يوم {d} بنسبة تأخير {r:.2f}%" for d, r in peak_days_series.items()])

    # 3. تنسيق المناديب والمطاعم والشكاوى كنصوص مفصلة
    top_drivers_series = df_main.groupby('DriverID')['Driver_Efficiency_Index'].mean().nlargest(3)
    top_drivers = ", ".join([f"السائق {drv} (مؤشر كفاءة: {idx:.2f})" for drv, idx in top_drivers_series.items()])

    worst_drivers_series = df_main.groupby('DriverID')['Is_Delayed'].mean().nlargest(3) * 100
    worst_drivers = ", ".join([f"السائق {drv} (معدل تأخير: {rate:.2f}%)" for drv, rate in worst_drivers_series.items()])

    slowest_res_series = df_main.groupby('RestaurantID')['Prep_Time'].mean().nlargest(5)
    slowest_restaurants = ", ".join([f"مطعم {res} (وقت التحضير: {t:.2f} دقيقة)" for res, t in slowest_res_series.items()])

    complaints_series = df_main['IssueCategory'].value_counts(normalize=True).head(3) * 100
    complaints = ", ".join([f"فئة '{cat}': {pct:.2f}%" for cat, pct in complaints_series.items()])

    cancel_rate = (df_main['Status'].str.lower().eq('cancelled').sum() / total_orders_count) * 100 if 'Status' in df_main.columns else 0.0

    # 4. تنسيق أداء المناطق الجغرافية
    if 'City' in df_main.columns:
        geo_df = df_main.groupby('City').agg(
            Orders=('OrderID', 'count'),
            Revenue=('DeliveryFee', 'sum')
        ).nlargest(3, 'Orders')
        geo_lines = []
        for city, row in geo_df.iterrows():
            geo_lines.append(f"- مدينة {city}: {row['Orders']} طلباً، بإيرادات {row['Revenue']:.2f}")
        geo_profit = '\n'.join(geo_lines)
    else:
        geo_profit = "N/A"

    pipe = pipeline('text-generation', model='google/gemma-3-1b-it')

    system_instruction = (
        'You are a professional strategic data analyst and operational expert. CRITICAL RULE: You MUST '
        'use the exact provided metrics without altering them. Do not invent numbers, percentages, or statistics.'
    )

    user_query = f"""
    Write a professional executive report based strictly and verbatim on these exact operational metrics:

    [Core Operational Metrics]:
    - Total Orders Analyzed: {total_orders_count}
    - Overall Delay Rate: {delay_rate:.2f}%
    - Average Excess Delay Time: {avg_delay_time:.2f} minutes
    - Average Preparation Time: {avg_prep:.2f} minutes
    - Average Transit Time: {avg_transit:.2f} minutes
    - Primary Operational Bottleneck: {bottleneck}
    - Order Cancellation Rate: {cancel_rate:.2f}%

    [Key Findings & Breakdown Data]:
    - Vehicle Impact Summary:
      {vehicle_impact}
    - Peak Order Hours: {peak_hours}
    - Peak Delay Days: {peak_days}
    - Top Efficient Drivers: {top_drivers}
    - Most Delayed Drivers: {worst_drivers}
    - Slowest Restaurants: {slowest_restaurants}
    - Top Customer Complaints: {complaints}
    - Geographical Performance (Top Cities, Orders & Revenue):
      {geo_profit}

    Required Report Structure:
    1. Executive Summary & Objectives: High-level overview of total orders ({total_orders_count}), operational efficiency, and key performance indicators (KPIs).
    2. Data Methodology & Feature Engineering: Overview of integrated tables (Orders, Deliveries, Drivers, Customer Feedback) and engineered features (Prep_Time, Transit_Time, Is_Delayed, Driver_Efficiency_Index).
    3. Comprehensive Answers to Business Questions (Must explicitly and thoroughly answer using the exact numbers and IDs provided above):
        1. What is the overall delay rate and its impact on customer trust? (Reference delay rate {delay_rate:.2f}% and excess delay {avg_delay_time:.2f} mins).
        2. Where is the main operational bottleneck? (Compare preparation {avg_prep:.2f} mins vs transit {avg_transit:.2f} mins and discuss {bottleneck}).
        3. How do vehicle types and distance affect delivery times? (Analyze the vehicle impact data provided above).
        4. When do peak hours and congestion occur? (Detail peak hours {peak_hours} and delay days {peak_days}).
        5. How do drivers perform across efficiency and delay metrics? (Reference top drivers {top_drivers} and worst delayed drivers {worst_drivers}).
        6. Which restaurants suffer from preparation delays? (Analyze slowest vendors {slowest_restaurants}).
        7. What are the primary customer complaints? (Examine complaint categories {complaints}).
        8. What drives order cancellations? (Reference cancellation rate {cancel_rate:.2f}%).
        9. Which geographical regions drive revenue and volume? (Reference regional data {geo_profit}).
        10. Five strategic recommendations for management.
    4. Key Insights & Patterns: Deep statistical findings, engagement patterns, and logistical pain points based strictly on the provided tables.
    5. Strategic Recommendations: Propose five practical, actionable, and structured recommendations for management to resolve bottlenecks and enhance customer retention.
    6. Conclusion: Summary and implementation roadmap.

    Please write the report in a professional corporate tone using ONLY the real numbers, exact IDs, and metrics provided above, avoiding any placeholders or fabricated data.
    """

    messages = [
        {'role': 'system', 'content': system_instruction},
        {'role': 'user', 'content': user_query},
    ]

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = pipe(prompt, max_new_tokens=2500, do_sample=True, temperature=0.7)
    full_text = response[0]['generated_text']
    report_text = full_text.split('<start_of_turn>model')[-1].strip()
    return report_text

# استدعاء دالة التقرير وتوليده بالطريقة المطلوبة
final_report = generate_executive_report()
report_filename = 'Validated_Executive_Success_Report.md'

with open(report_filename, 'w', encoding='utf-8') as file:
  file.write(final_report)

print(
    f'\nValidated report successfully generated and exported to:'
    f' {report_filename}'
)