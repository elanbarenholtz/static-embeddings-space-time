"""Activation extraction for the 272 cities (run on a GPU; originally a notebook). The analyses
reported in the paper use city_resource/pythia_cities.py on the saved activations."""
import torch, numpy as np, pandas as pd
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())

# ----
CITIES = [["anchorage", 61.22, -149.9, -2.5, "North America", 1914, 30, 76300, 291000], ["fairbanks", 64.84, -147.72, -2.4, "North America", 1901, 136, 76300, 32000], ["juneau", 58.3, -134.42, 4.8, "North America", 1881, 17, 76300, 32000], ["vancouver", 49.28, -123.12, 10.4, "North America", 1886, 0, 52100, 631000], ["seattle", 47.61, -122.33, 11.3, "North America", 1851, 53, 76300, 737000], ["portland", 45.52, -122.68, 12.4, "North America", 1845, 15, 76300, 652000], ["san francisco", 37.77, -122.42, 14.6, "North America", 1776, 16, 76300, 874000], ["los angeles", 34.05, -118.24, 18.3, "North America", 1781, 71, 76300, 3900000], ["san diego", 32.72, -117.16, 17.8, "North America", 1769, 20, 76300, 1420000], ["las vegas", 36.17, -115.14, 20.3, "North America", 1905, 620, 76300, 641000], ["phoenix", 33.45, -112.07, 24.2, "North America", 1867, 331, 76300, 1608000], ["salt lake city", 40.76, -111.89, 11.5, "North America", 1847, 1288, 76300, 200000], ["denver", 39.74, -104.99, 10.4, "North America", 1858, 1609, 76300, 715000], ["albuquerque", 35.08, -106.65, 14.0, "North America", 1706, 1520, 76300, 564000], ["el paso", 31.76, -106.49, 18.4, "North America", 1659, 1140, 76300, 678000], ["dallas", 32.78, -96.8, 18.9, "North America", 1841, 131, 76300, 1304000], ["houston", 29.76, -95.37, 20.7, "North America", 1836, 15, 76300, 2300000], ["san antonio", 29.42, -98.49, 20.3, "North America", 1718, 198, 76300, 1434000], ["austin", 30.27, -97.74, 20.6, "North America", 1839, 149, 76300, 964000], ["new orleans", 29.95, -90.07, 20.7, "North America", 1718, 0, 76300, 383000], ["minneapolis", 44.98, -93.27, 7.8, "North America", 1856, 264, 76300, 429000], ["chicago", 41.88, -87.63, 10.0, "North America", 1833, 176, 76300, 2697000], ["detroit", 42.33, -83.05, 10.1, "North America", 1701, 183, 76300, 639000], ["indianapolis", 39.77, -86.16, 11.8, "North America", 1821, 218, 76300, 887000], ["columbus", 39.96, -82.99, 11.6, "North America", 1812, 275, 76300, 906000], ["cleveland", 41.5, -81.69, 10.6, "North America", 1796, 199, 76300, 373000], ["pittsburgh", 40.44, -79.99, 10.8, "North America", 1758, 230, 76300, 303000], ["nashville", 36.16, -86.78, 15.4, "North America", 1779, 169, 76300, 689000], ["memphis", 35.15, -90.05, 16.8, "North America", 1819, 84, 76300, 633000], ["atlanta", 33.75, -84.39, 16.9, "North America", 1837, 320, 76300, 499000], ["miami", 25.76, -80.19, 25.0, "North America", 1896, 2, 76300, 442000], ["tampa", 27.95, -82.46, 22.8, "North America", 1849, 14, 76300, 384000], ["orlando", 28.54, -81.38, 22.4, "North America", 1875, 34, 76300, 307000], ["jacksonville", 30.33, -81.66, 20.7, "North America", 1791, 4, 76300, 949000], ["charlotte", 35.23, -80.84, 16.0, "North America", 1768, 229, 76300, 874000], ["washington", 38.91, -77.04, 14.6, "North America", 1790, 22, 76300, 689000], ["baltimore", 39.29, -76.61, 13.1, "North America", 1729, 10, 76300, 576000], ["philadelphia", 39.95, -75.17, 12.8, "North America", 1682, 12, 76300, 1576000], ["new york", 40.71, -74.01, 12.9, "North America", 1624, 10, 76300, 8336000], ["newark", 40.74, -74.17, 12.6, "North America", 1666, 10, 76300, 311000], ["boston", 42.36, -71.06, 10.9, "North America", 1630, 6, 76300, 675000], ["montreal", 45.5, -73.57, 6.8, "North America", 1642, 36, 52100, 1762000], ["toronto", 43.65, -79.38, 9.4, "North America", 1793, 76, 52100, 2794000], ["ottawa", 45.42, -75.7, 6.6, "North America", 1826, 70, 52100, 1000000], ["calgary", 51.05, -114.07, 4.4, "North America", 1884, 1045, 52100, 1336000], ["edmonton", 53.55, -113.49, 2.7, "North America", 1795, 668, 52100, 1010000], ["winnipeg", 49.9, -97.14, 3.0, "North America", 1862, 238, 52100, 749000], ["quebec city", 46.81, -71.21, 4.9, "North America", 1608, 50, 52100, 549000], ["halifax", 44.65, -63.57, 7.0, "North America", 1749, 10, 52100, 439000], ["mexico city", 19.43, -99.13, 16.0, "North America", 1325, 2240, 10400, 9210000], ["guadalajara", 20.67, -103.35, 19.6, "North America", 1542, 1566, 10400, 1495000], ["monterrey", 25.67, -100.31, 22.1, "North America", 1596, 540, 10400, 1135000], ["havana", 23.11, -82.37, 25.2, "North America", 1519, 30, 9500, 2130000], ["kingston", 18.0, -76.79, 27.0, "North America", 1692, 9, 5500, 670000], ["san jose cr", 9.93, -84.08, 20.5, "North America", 1738, 1161, 13200, 353000], ["panama city", 8.98, -79.52, 27.3, "North America", 1519, 2, 17100, 880000], ["guatemala city", 14.63, -90.51, 19.5, "North America", 1776, 1500, 5000, 1000000], ["tegucigalpa", 14.07, -87.21, 21.5, "North America", 1578, 990, 2800, 1160000], ["managua", 12.13, -86.25, 27.1, "North America", 1819, 83, 2000, 1055000], ["bogota", 4.71, -74.07, 14.0, "South America", 1538, 2640, 6500, 7410000], ["medellin", 6.25, -75.56, 22.0, "South America", 1616, 1495, 6500, 2530000], ["quito", -0.18, -78.47, 13.5, "South America", 1534, 2850, 6100, 1980000], ["lima", -12.05, -77.04, 19.2, "South America", 1535, 154, 7000, 9750000], ["la paz", -16.5, -68.15, 8.5, "South America", 1548, 3640, 3600, 812000], ["santiago", -33.45, -70.67, 14.4, "South America", 1541, 520, 16300, 5614000], ["buenos aires", -34.6, -58.38, 17.9, "South America", 1536, 25, 13700, 3075000], ["montevideo", -34.88, -56.16, 16.6, "South America", 1724, 43, 17300, 1382000], ["asuncion", -25.26, -57.58, 23.0, "South America", 1537, 64, 5900, 524000], ["brasilia", -15.79, -47.88, 21.3, "South America", 1960, 1172, 8900, 3055000], ["sao paulo", -23.55, -46.63, 19.2, "South America", 1554, 760, 8900, 12330000], ["rio de janeiro", -22.91, -43.17, 23.8, "South America", 1565, 11, 8900, 6748000], ["caracas", 10.48, -66.9, 22.0, "South America", 1567, 900, 3700, 2940000], ["georgetown", 6.8, -58.16, 27.0, "South America", 1781, 1, 16300, 200000], ["paramaribo", 5.85, -55.2, 27.0, "South America", 1613, 3, 6700, 240000], ["recife", -8.05, -34.87, 25.8, "South America", 1537, 4, 8900, 1654000], ["salvador", -12.97, -38.51, 25.3, "South America", 1549, 8, 8900, 2886000], ["manaus", -3.12, -60.02, 27.4, "South America", 1669, 92, 8900, 2220000], ["curitiba", -25.43, -49.27, 17.1, "South America", 1693, 934, 8900, 1948000], ["cordoba", -31.42, -64.18, 18.0, "South America", 1573, 390, 13700, 1391000], ["reykjavik", 64.15, -21.94, 4.7, "Europe", 874, 14, 68100, 131000], ["oslo", 59.91, 10.75, 6.3, "Europe", 1040, 23, 82800, 697000], ["stockholm", 59.33, 18.07, 6.6, "Europe", 1252, 28, 55600, 975000], ["helsinki", 60.17, 24.94, 5.9, "Europe", 1550, 8, 49200, 658000], ["tallinn", 59.44, 24.75, 6.0, "Europe", 1248, 40, 27700, 450000], ["riga", 56.95, 24.11, 6.4, "Europe", 1201, 7, 21200, 615000], ["vilnius", 54.69, 25.28, 6.7, "Europe", 1323, 112, 23500, 580000], ["copenhagen", 55.68, 12.57, 8.7, "Europe", 1167, 14, 63000, 794000], ["dublin", 53.35, -6.26, 9.8, "Europe", 841, 20, 100200, 554000], ["edinburgh", 55.95, -3.19, 8.9, "Europe", 1125, 47, 46500, 525000], ["london", 51.51, -0.13, 11.3, "Europe", 43, 11, 46500, 8982000], ["amsterdam", 52.37, 4.9, 10.2, "Europe", 1275, -2, 57100, 873000], ["brussels", 50.85, 4.35, 10.5, "Europe", 979, 13, 51800, 185000], ["paris", 48.86, 2.35, 11.3, "Europe", -250, 35, 42300, 2161000], ["berlin", 52.52, 13.41, 9.6, "Europe", 1237, 34, 51200, 3645000], ["hamburg", 53.55, 9.99, 9.0, "Europe", 810, 6, 51200, 1845000], ["munich", 48.14, 11.58, 8.7, "Europe", 1158, 520, 51200, 1472000], ["frankfurt", 50.11, 8.68, 10.4, "Europe", 794, 112, 51200, 753000], ["zurich", 47.38, 8.54, 9.3, "Europe", -15, 408, 93500, 421000], ["geneva", 46.2, 6.14, 10.5, "Europe", -500, 375, 93500, 203000], ["vienna", 48.21, 16.37, 10.4, "Europe", -500, 171, 53600, 1911000], ["prague", 50.08, 14.44, 8.4, "Europe", 885, 235, 27200, 1309000], ["warsaw", 52.23, 21.01, 8.5, "Europe", 1300, 100, 17800, 1794000], ["krakow", 50.06, 19.94, 8.5, "Europe", 965, 219, 17800, 780000], ["budapest", 47.5, 19.04, 11.3, "Europe", 1873, 96, 18900, 1752000], ["bucharest", 44.43, 26.1, 10.8, "Europe", 1459, 55, 14800, 1794000], ["sofia", 42.7, 23.32, 10.6, "Europe", -500, 550, 12200, 1241000], ["belgrade", 44.79, 20.47, 11.8, "Europe", -279, 117, 8900, 1166000], ["zagreb", 45.81, 15.98, 11.3, "Europe", 1094, 122, 17400, 790000], ["ljubljana", 46.06, 14.51, 10.4, "Europe", -2000, 295, 28400, 286000], ["rome", 41.9, 12.5, 15.7, "Europe", -753, 21, 34500, 2873000], ["milan", 45.46, 9.19, 13.1, "Europe", -400, 120, 34500, 1372000], ["naples", 40.85, 14.27, 16.0, "Europe", -470, 17, 34500, 967000], ["florence", 43.77, 11.25, 14.6, "Europe", -59, 50, 34500, 382000], ["venice", 45.44, 12.32, 13.0, "Europe", 421, 1, 34500, 261000], ["madrid", 40.42, -3.7, 14.5, "Europe", 852, 650, 30100, 3223000], ["barcelona", 41.39, 2.17, 15.5, "Europe", -230, 12, 30100, 1620000], ["seville", 37.39, -5.99, 18.6, "Europe", -206, 7, 30100, 688000], ["lisbon", 38.72, -9.14, 16.3, "Europe", -1200, 2, 24500, 545000], ["porto", 41.15, -8.61, 14.4, "Europe", -300, 85, 24500, 249000], ["athens", 37.98, 23.73, 17.8, "Europe", -3000, 70, 20200, 664000], ["thessaloniki", 40.64, 22.94, 15.8, "Europe", -315, 5, 20200, 325000], ["marseille", 43.3, 5.37, 15.0, "Europe", -600, 12, 42300, 870000], ["lyon", 45.76, 4.84, 12.4, "Europe", -43, 175, 42300, 516000], ["minsk", 53.9, 27.57, 6.7, "Europe", 1067, 220, 7000, 1996000], ["kyiv", 50.45, 30.52, 8.4, "Europe", 482, 179, 4800, 2952000], ["moscow", 55.76, 37.62, 5.8, "Europe", 1147, 156, 12100, 12636000], ["saint petersburg", 59.93, 30.32, 5.8, "Europe", 1703, 3, 12100, 5384000], ["cairo", 30.04, 31.24, 21.4, "Africa", 969, 75, 3900, 9500000], ["alexandria", 31.2, 29.92, 20.5, "Africa", -331, 5, 3900, 5200000], ["casablanca", 33.59, -7.59, 17.7, "Africa", 768, 27, 3600, 3360000], ["marrakech", 31.63, -8.01, 19.6, "Africa", 1062, 457, 3600, 929000], ["tunis", 36.81, 10.18, 18.4, "Africa", -814, 4, 3800, 639000], ["algiers", 36.75, 3.04, 17.6, "Africa", -944, 0, 4000, 3415000], ["tripoli", 32.9, 13.18, 20.0, "Africa", -700, 7, 6100, 1170000], ["lagos", 6.52, 3.38, 27.0, "Africa", 1472, 41, 2100, 9000000], ["abuja", 9.06, 7.49, 25.7, "Africa", 1828, 476, 2100, 1236000], ["accra", 5.56, -0.19, 27.1, "Africa", 1877, 61, 2400, 2291000], ["dakar", 14.69, -17.44, 25.0, "Africa", 1857, 22, 1600, 1146000], ["bamako", 12.65, -8.0, 28.2, "Africa", 1640, 350, 900, 2446000], ["abidjan", 5.36, -4.01, 26.4, "Africa", 1898, 18, 2500, 4707000], ["kinshasa", -4.44, 15.27, 25.3, "Africa", 1881, 240, 580, 14970000], ["brazzaville", -4.27, 15.28, 25.0, "Africa", 1880, 317, 2400, 1838000], ["luanda", -8.84, 13.23, 25.8, "Africa", 1575, 6, 3500, 2776000], ["nairobi", -1.29, 36.82, 17.8, "Africa", 1899, 1795, 2100, 4397000], ["kampala", 0.35, 32.58, 21.5, "Africa", 1890, 1190, 860, 1650000], ["dar es salaam", -6.79, 39.28, 26.0, "Africa", 1862, 24, 1100, 4365000], ["addis ababa", 9.02, 38.75, 16.3, "Africa", 1886, 2355, 960, 3352000], ["khartoum", 15.5, 32.56, 29.9, "Africa", 1821, 382, 600, 1975000], ["mogadishu", 2.05, 45.34, 27.1, "Africa", 900, 12, 500, 2590000], ["johannesburg", -26.2, 28.04, 16.0, "Africa", 1886, 1753, 6000, 957000], ["cape town", -33.93, 18.42, 16.7, "Africa", 1652, 0, 6000, 433000], ["durban", -29.86, 31.02, 20.5, "Africa", 1835, 8, 6000, 595000], ["pretoria", -25.75, 28.19, 17.5, "Africa", 1855, 1339, 6000, 742000], ["harare", -17.83, 31.05, 18.0, "Africa", 1890, 1490, 1200, 1530000], ["lusaka", -15.39, 28.32, 20.5, "Africa", 1905, 1280, 1200, 2526000], ["maputo", -25.97, 32.57, 22.8, "Africa", 1781, 47, 500, 1102000], ["antananarivo", -18.91, 47.52, 18.4, "Africa", 1625, 1276, 530, 1275000], ["windhoek", -22.56, 17.08, 20.0, "Africa", 1840, 1655, 4900, 431000], ["gaborone", -24.63, 25.91, 21.0, "Africa", 1964, 1010, 7000, 232000], ["douala", 4.05, 9.77, 26.7, "Africa", 1884, 13, 1700, 2768000], ["yaounde", 3.87, 11.52, 23.8, "Africa", 1888, 726, 1700, 2765000], ["conakry", 9.64, -13.58, 26.8, "Africa", 1884, 6, 1300, 1668000], ["freetown", 8.48, -13.23, 26.6, "Africa", 1792, 14, 500, 1056000], ["istanbul", 41.01, 28.98, 14.1, "Asia", -660, 39, 10600, 15460000], ["ankara", 39.93, 32.87, 12.0, "Asia", -1200, 938, 10600, 5663000], ["tehran", 35.69, 51.39, 17.0, "Asia", -3000, 1191, 3800, 8694000], ["baghdad", 33.31, 44.37, 22.8, "Asia", -762, 34, 5000, 7144000], ["riyadh", 24.71, 46.68, 26.0, "Asia", 1737, 612, 23200, 7676000], ["jeddah", 21.49, 39.19, 28.5, "Asia", 647, 12, 23200, 4076000], ["dubai", 25.2, 55.27, 27.2, "Asia", 1833, 5, 43100, 3331000], ["doha", 25.29, 51.53, 27.1, "Asia", 1825, 10, 66000, 1186000], ["muscat", 23.59, 58.54, 28.3, "Asia", -2000, 5, 19300, 797000], ["kuwait city", 29.38, 47.99, 26.3, "Asia", 1613, 5, 32000, 510000], ["beirut", 33.89, 35.5, 20.4, "Asia", -3000, 0, 4100, 361000], ["damascus", 33.51, 36.29, 17.0, "Asia", -8000, 680, 500, 1711000], ["amman", 31.95, 35.93, 17.5, "Asia", -7000, 757, 4200, 1148000], ["jerusalem", 31.77, 35.23, 17.4, "Asia", -4000, 754, 51400, 936000], ["tel aviv", 32.08, 34.78, 20.1, "Asia", 1909, 5, 51400, 460000], ["karachi", 24.86, 67.01, 26.0, "Asia", 1729, 14, 1500, 14910000], ["lahore", 31.55, 74.35, 24.3, "Asia", -2000, 217, 1500, 11130000], ["islamabad", 33.69, 73.04, 21.3, "Asia", 1960, 507, 1500, 1015000], ["kabul", 34.53, 69.17, 12.1, "Asia", -1500, 1791, 500, 4222000], ["new delhi", 28.61, 77.21, 25.2, "Asia", 1911, 216, 2400, 11035000], ["mumbai", 19.08, 72.88, 27.2, "Asia", 1507, 14, 2400, 12442000], ["kolkata", 22.57, 88.36, 26.8, "Asia", 1690, 9, 2400, 4497000], ["chennai", 13.08, 80.27, 28.6, "Asia", 1639, 6, 2400, 7088000], ["bangalore", 12.97, 77.59, 24.0, "Asia", 1537, 920, 2400, 8443000], ["hyderabad", 17.39, 78.49, 26.6, "Asia", 1591, 542, 2400, 6810000], ["dhaka", 23.81, 90.41, 25.7, "Asia", 1608, 4, 2500, 8907000], ["colombo", 6.93, 79.84, 27.5, "Asia", -500, 1, 3900, 752000], ["kathmandu", 27.72, 85.32, 18.1, "Asia", 723, 1400, 1200, 1003000], ["yangon", 16.87, 96.2, 27.4, "Asia", 1755, 4, 1200, 5160000], ["bangkok", 13.76, 100.5, 28.6, "Asia", 1782, 2, 7200, 5676000], ["hanoi", 21.03, 105.85, 23.5, "Asia", 1010, 12, 3700, 8054000], ["ho chi minh city", 10.82, 106.63, 27.8, "Asia", 1698, 19, 3700, 8993000], ["phnom penh", 11.56, 104.92, 28.3, "Asia", 1434, 12, 1700, 2129000], ["kuala lumpur", 3.14, 101.69, 27.7, "Asia", 1857, 22, 11400, 1982000], ["singapore", 1.35, 103.82, 27.4, "Asia", 1819, 15, 65200, 5454000], ["jakarta", -6.21, 106.85, 27.0, "Asia", 1527, 8, 4300, 10560000], ["manila", 14.6, 120.98, 28.4, "Asia", 1571, 5, 3500, 1846000], ["beijing", 39.9, 116.4, 12.9, "Asia", -1045, 43, 12600, 21540000], ["shanghai", 31.23, 121.47, 16.1, "Asia", 1074, 4, 12600, 24870000], ["guangzhou", 23.13, 113.26, 22.0, "Asia", -214, 11, 12600, 18680000], ["shenzhen", 22.54, 114.06, 22.5, "Asia", 1979, 0, 12600, 17560000], ["hong kong", 22.32, 114.17, 23.3, "Asia", 1841, 32, 49800, 7413000], ["chengdu", 30.57, 104.07, 16.3, "Asia", -311, 500, 12600, 16330000], ["wuhan", 30.59, 114.31, 16.8, "Asia", -1500, 37, 12600, 12327000], ["chongqing", 29.56, 106.55, 18.3, "Asia", -316, 244, 12600, 8700000], ["taipei", 25.03, 121.57, 22.7, "Asia", 1709, 9, 32700, 2602000], ["seoul", 37.57, 126.98, 12.5, "Asia", -18, 38, 33200, 9776000], ["busan", 35.18, 129.08, 14.7, "Asia", 1876, 20, 33200, 3449000], ["pyongyang", 39.02, 125.74, 10.8, "Asia", -1122, 27, 1800, 3255000], ["tokyo", 35.68, 139.69, 15.4, "Asia", 1457, 40, 33800, 13960000], ["osaka", 34.69, 135.5, 16.9, "Asia", -645, 5, 33800, 2753000], ["sapporo", 43.06, 141.35, 8.9, "Asia", 1868, 29, 33800, 1973000], ["ulaanbaatar", 47.91, 106.91, -0.4, "Asia", 1639, 1350, 4500, 1540000], ["novosibirsk", 55.01, 82.93, 1.8, "Asia", 1893, 152, 12100, 1612000], ["vladivostok", 43.12, 131.87, 5.0, "Asia", 1860, 8, 12100, 605000], ["almaty", 43.24, 76.95, 10.0, "Asia", 1854, 776, 10100, 1916000], ["tashkent", 41.3, 69.28, 14.2, "Asia", -200, 455, 1900, 2571000], ["tbilisi", 41.69, 44.8, 13.3, "Asia", 458, 380, 5000, 1118000], ["yerevan", 40.18, 44.51, 12.4, "Asia", -782, 990, 4700, 1075000], ["baku", 40.41, 49.87, 14.2, "Asia", -3000, 0, 5800, 2262000], ["sydney", -33.87, 151.21, 17.7, "Oceania", 1788, 3, 55100, 5312000], ["melbourne", -37.81, 144.96, 14.5, "Oceania", 1835, 31, 55100, 5078000], ["brisbane", -27.47, 153.03, 20.4, "Oceania", 1825, 5, 55100, 2514000], ["perth", -31.95, 115.86, 18.5, "Oceania", 1829, 0, 55100, 2085000], ["adelaide", -34.93, 138.6, 16.7, "Oceania", 1836, 50, 55100, 1346000], ["auckland", -36.85, 174.76, 15.1, "Oceania", 1840, 0, 47300, 1657000], ["wellington", -41.29, 174.78, 12.8, "Oceania", 1840, 0, 47300, 215000], ["christchurch", -43.53, 172.64, 11.7, "Oceania", 1850, 20, 47300, 380000], ["darwin", -12.46, 130.84, 27.6, "Oceania", 1869, 37, 55100, 148000], ["hobart", -42.88, 147.33, 12.0, "Oceania", 1804, 10, 55100, 240000], ["canberra", -35.28, 149.13, 13.1, "Oceania", 1913, 577, 55100, 457000], ["suva", -18.14, 178.44, 25.6, "Oceania", 1849, 6, 6100, 93000], ["port moresby", -6.21, 147.0, 26.9, "Oceania", 1873, 35, 2800, 365000], ["omaha", 41.26, -95.94, 10.6, "North America", 1854, 332, 76300, 486000], ["kansas city", 39.1, -94.58, 12.5, "North America", 1838, 247, 76300, 508000], ["st louis", 38.63, -90.2, 13.3, "North America", 1764, 142, 76300, 302000], ["milwaukee", 43.04, -87.91, 8.6, "North America", 1846, 188, 76300, 577000], ["cincinnati", 39.1, -84.51, 12.5, "North America", 1788, 264, 76300, 309000], ["richmond", 37.54, -77.44, 14.2, "North America", 1737, 63, 76300, 226000], ["raleigh", 35.78, -78.64, 15.5, "North America", 1792, 96, 76300, 468000], ["birmingham", 33.52, -86.8, 17.1, "North America", 1871, 196, 76300, 200000], ["oklahoma city", 35.47, -97.52, 15.9, "North America", 1889, 366, 76300, 681000], ["tucson", 32.22, -110.93, 22.0, "North America", 1775, 728, 76300, 542000], ["honolulu", 21.31, -157.86, 25.1, "North America", 1907, 5, 76300, 350000], ["nice", 43.71, 7.26, 15.5, "Europe", -350, 10, 42300, 342000], ["palermo", 38.12, 13.36, 18.0, "Europe", -734, 14, 34500, 663000], ["bologna", 44.49, 11.34, 13.6, "Europe", -510, 54, 34500, 394000], ["gdansk", 54.35, 18.65, 7.9, "Europe", 997, 7, 17800, 471000], ["bratislava", 48.15, 17.11, 10.3, "Europe", -400, 134, 21200, 475000], ["nicosia", 35.19, 33.38, 19.5, "Europe", -2500, 149, 31000, 326000], ["valletta", 35.9, 14.51, 19.0, "Europe", 1566, 15, 32200, 6000], ["tirana", 41.33, 19.82, 15.2, "Europe", 1614, 110, 6300, 418000], ["skopje", 42.0, 21.43, 12.4, "Europe", -400, 240, 6700, 544000], ["sarajevo", 43.86, 18.41, 10.1, "Europe", 1461, 518, 7000, 275000], ["podgorica", 42.44, 19.26, 15.3, "Europe", 1326, 44, 9500, 187000], ["nanjing", 32.06, 118.8, 15.7, "Asia", -495, 20, 12600, 9314000], ["xian", 34.26, 108.94, 14.1, "Asia", -1100, 405, 12600, 8700000], ["hangzhou", 30.27, 120.15, 16.6, "Asia", -2200, 19, 12600, 10360000], ["surabaya", -7.25, 112.75, 27.3, "Asia", 1293, 5, 4300, 2874000], ["bandung", -6.91, 107.61, 23.5, "Asia", 1810, 768, 4300, 2444000], ["nagoya", 35.18, 136.91, 15.8, "Asia", 1610, 17, 33800, 2296000], ["fukuoka", 33.59, 130.4, 17.0, "Asia", 57, 3, 33800, 1603000], ["pune", 18.52, 73.86, 25.0, "Asia", 847, 560, 2400, 3124000], ["ahmedabad", 23.02, 72.57, 27.0, "Asia", 1411, 55, 2400, 5577000], ["jaipur", 26.92, 75.79, 25.5, "Asia", 1727, 431, 2400, 3046000], ["lucknow", 26.85, 80.95, 25.8, "Asia", 1775, 123, 2400, 2815000], ["chittagong", 22.36, 91.78, 25.7, "Asia", 1340, 9, 2500, 2582000], ["chiang mai", 18.79, 98.98, 25.4, "Asia", 1296, 310, 7200, 131000], ["vientiane", 17.97, 102.63, 26.5, "Asia", 1560, 174, 2600, 820000]]

print(len(CITIES), 'cities')

# ----
import gensim.downloader as api
print('loading GloVe...',    flush=True); g = api.load('glove-wiki-gigaword-300')
print('loading Word2Vec...', flush=True); w = api.load('word2vec-google-news-300')

def gv(nm):
    v = [g[t] for t in nm.lower().split() if t in g]
    return np.mean(v, 0) if v else None

def wv(nm):
    ph = '_'.join(t.capitalize() for t in nm.split())
    if ph in w: return np.asarray(w[ph], dtype=np.float64)
    f = [t.capitalize() if t.capitalize() in w else t.lower()
         for t in nm.split() if t.capitalize() in w or t.lower() in w]
    return np.mean([w[t] for t in f], 0).astype(np.float64) if f else None

KEEP = [c for c in CITIES if gv(c[0]) is not None and wv(c[0]) is not None]
STATIC = {'GloVe':    np.array([gv(c[0]) for c in KEEP], dtype=np.float64),
          'Word2Vec': np.array([wv(c[0]) for c in KEEP], dtype=np.float64)}
print(f'{len(KEEP)}/{len(CITIES)} cities have both static vectors')

# ----
MODEL = 'EleutherAI/pythia-2.8b'   # <-- match whichever run you are explaining
TEMPLATE = '{}'
LAYER_STRIDE = 1                   # 2 halves the runtime on a deep model

# ----
from transformers import AutoTokenizer, AutoModelForCausalLM

dev   = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if dev == 'cuda' else torch.float32
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
print(f'{model.config.num_hidden_layers} layers, hidden {model.config.hidden_size}')

disp = [' '.join(t.capitalize() for t in c[0].split()) for c in KEEP]
acts = None
with torch.no_grad():
    for i, nm in enumerate(disp):
        ids = tok(TEMPLATE.format(nm), return_tensors='pt').to(dev)
        hs  = model(**ids, output_hidden_states=True).hidden_states
        if acts is None:
            acts = np.zeros((len(hs), len(disp), hs[0].shape[-1]), dtype=np.float32)
        for L in range(len(hs)):
            acts[L, i] = hs[L][0, -1].float().cpu().numpy()
        if (i + 1) % 50 == 0: print(f'  {i+1}/{len(disp)}', flush=True)

tag = MODEL.split('/')[-1]
np.save(f'acts_{tag}.npy', acts)        # <-- KEEP THIS. Everything else is cheap to redo.
print('activations', acts.shape, '-> saved to', f'acts_{tag}.npy')

# ----
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import r2_score

ALPHAS = np.logspace(-2, 3, 20)
n      = len(KEEP)
lat    = np.array([c[1] for c in KEEP], float)
lon    = np.array([c[2] for c in KEEP], float)
temp   = np.array([c[3] for c in KEEP], float)
cont   = [c[4] for c in KEEP]
conts  = sorted(set(cont)); cid = np.array([conts.index(c) for c in cont])
targets = {'Latitude': lat, 'Longitude': lon, 'Temperature': temp}

rand_folds = [train_test_split(np.arange(n), test_size=.2, random_state=s) for s in range(10)]
cont_folds = list(LeaveOneGroupOut().split(np.arange(n), np.zeros(n), cid))
fit = lambda X, y: RidgeCV(alphas=ALPHAS).fit(X, y)
print(f'{n} cities, {len(conts)} continents')

# ----
from sklearn.linear_model import RidgeCV as RCV
rng  = np.random.default_rng(0)
PERM = rng.permutation(n)                     # one fixed shuffle, used throughout
RESID_ALPHAS = np.logspace(0, 5, 12)          # the removal step is heavily regularised
STATIC_USE   = ['GloVe', 'Word2Vec']          # ['GloVe'] alone runs in half the time

def split_parts(A, S, tr):
    # (projection, residual) for ALL rows; the map S->A is fit on training rows only
    P = RCV(alphas=RESID_ALPHAS).fit(S[tr], A[tr]).predict(S)
    return P, A - P

def pca_matched_resid(A, target_frac):
    # remove the top principal directions of A until the variance removed matches
    # `target_frac`. Target-blind, but variance-matched to the true removal -- this is
    # the null that answers "any removal this large would have done it".
    Ac = A - A.mean(0)
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    frac = np.cumsum(s**2) / np.sum(s**2)
    j = int(np.searchsorted(frac, target_frac) + 1)
    j = max(1, min(j, Vt.shape[0]))
    Q = Vt[:j].T
    return A - (A @ Q) @ Q.T, j

def rand_subspace_resid(A, k, seed):
    # remove a random k-dimensional subspace of activation space: the dimensionality-
    # matched null. Nothing about the cities enters, so anything it removes is generic.
    G = np.random.default_rng(seed).normal(size=(A.shape[1], k))
    Q = np.linalg.qr(G)[0]
    return A - (A @ Q) @ Q.T

def var_removed(A, R):
    # fraction of the activations' total variance that the removal took out
    Ac = A - A.mean(0)
    return float(1 - (np.sum((R - R.mean(0))**2) / np.sum(Ac**2)))

def probe(F, y, folds):
    return float(np.mean([r2_score(y[te], fit(F[tr], y[tr]).predict(F[te])) for tr, te in folds]))

def probe_pooled(F, y, folds):
    pred = np.empty(n)
    for tr, te in folds: pred[te] = fit(F[tr], y[tr]).predict(F[te])
    return float(r2_score(y, pred))

# ----
rows = []
for L in range(0, acts.shape[0], LAYER_STRIDE):
    A = acts[L].astype(np.float64)
    for sname in STATIC_USE:
        S, Sp = STATIC[sname], STATIC[sname][PERM]
        Arand = rand_subspace_resid(A, S.shape[1], seed=1000 + L)
        _pre  = [split_parts(A, S, tr) for tr, _ in rand_folds]
        vr_true = float(np.mean([var_removed(A, R) for _, R in _pre]))
        Apca, n_pcs = pca_matched_resid(A, vr_true)      # variance-matched null
        parts_rand = [(tr, te) + split_parts(A, S, tr) + split_parts(A, Sp, tr) + (Arand, Apca)
                      for tr, te in rand_folds]
        parts_cont = [(tr, te) + split_parts(A, S, tr) + split_parts(A, Sp, tr) + (Arand, Apca)
                      for tr, te in cont_folds]
        # p = (tr, te, proj, resid, proj_shuf, resid_shuf, resid_randsub, resid_pca)
        COND = [('full', None), ('proj', 2), ('resid', 3),
                ('resid_pcamatched', 7), ('resid_randsub', 6), ('resid_shuffled', 5)]
        vr_shuf = float(np.mean([var_removed(A, p[5]) for p in parts_rand]))
        vr_rand = var_removed(A, Arand)
        vr_pca  = var_removed(A, Apca)
        for tn, y in targets.items():
            out = {}
            for key, idx in COND:
                v = [r2_score(y[p[1]],
                              fit((A if idx is None else p[idx])[p[0]], y[p[0]])
                              .predict((A if idx is None else p[idx])[p[1]]))
                     for p in parts_rand]
                out[key] = float(np.mean(v)); out[key + '_sd'] = float(np.std(v))
                pred = np.empty(n)
                for p in parts_cont:
                    F = A if idx is None else p[idx]
                    pred[p[1]] = fit(F[p[0]], y[p[0]]).predict(F[p[1]])
                out[key + '_cont'] = float(r2_score(y, pred))
            rows.append(dict(model=MODEL, layer=L, static=sname, target=tn,
                             var_removed_true=vr_true, var_removed_shuffled=vr_shuf,
                             var_removed_randsub=vr_rand, var_removed_pcamatched=vr_pca,
                             n_pcs_removed=n_pcs, **out))
        r = {x['target']: x for x in rows if x['layer'] == L and x['static'] == sname}
        print(f"L{L:3d} {sname:9s} | " + '  '.join(
            f"{t[:4]} full {r[t]['full']:+.3f} resid {r[t]['resid']:+.3f} "
            f"(pca-matched {r[t]['resid_pcamatched']:+.3f})" for t in targets)
            + f"   [var removed: true {vr_true:.2f}, pca {vr_pca:.2f} ({n_pcs} PCs), "
              f"randsub {vr_rand:.2f}, shuf {vr_shuf:.2f}]",
            flush=True)

df  = pd.DataFrame(rows)
tag = MODEL.split('/')[-1]
df.to_csv(f'residualise_{tag}.csv', index=False)
df.head()

# ----
import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
 'axes.facecolor':'white','savefig.facecolor':'white','font.size':10})

print(f'{"target":12s} {"static":9s} {"L*":>4s} {"full":>7s} {"proj":>7s} '
      f'{"resid":>7s} {"pca-null":>9s} {"randsub":>8s}  {"kept":>6s}')
for sname in STATIC:
    for tn in targets:
        s  = df[(df.static == sname) & (df.target == tn)]
        b  = s.loc[s.full.idxmax()]
        kept = b.resid / b.full if b.full > 0 else float('nan')
        print(f'{tn:12s} {sname:9s} {int(b.layer):4d} {b.full:+7.3f} {b.proj:+7.3f} '
              f'{b.resid:+7.3f} {b.resid_pcamatched:+9.3f} {b.resid_randsub:+8.3f}  {100*kept:5.0f}%')

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
for row, sname in enumerate(STATIC):
    for ax, tn in zip(axes[row], targets):
        s = df[(df.static == sname) & (df.target == tn)].sort_values('layer')
        ax.plot(s.layer, s.full,       '-o', ms=3, lw=1.8, color='#2563eb', label='full activations')
        ax.plot(s.layer, s.proj,       '-',  lw=1.5, color='#16a34a', label=f'part explained by {sname}')
        ax.plot(s.layer, s.resid,      '-s', ms=3, lw=1.8, color='#dc2626', label=f'residual after removing {sname}')
        ax.plot(s.layer, s.resid_pcamatched, '--', lw=1.4, color='#94a3b8', label='residual, variance-matched PCA null')
        ax.plot(s.layer, s.resid_randsub, ':', lw=1.2, color='#a855f7', label='residual, random 300-d subspace')
        ax.axhline(0, color='#94a3b8', lw=1)
        ax.set_title(f'{tn} — {sname}', loc='left', fontsize=10)
        ax.set_xlabel('layer'); ax.set_ylim(-0.6, 1.0)
    axes[row][0].set_ylabel('probe $R^2$ (random 80/20)')
axes[0][0].legend(frameon=False, fontsize=7.5, loc='lower right')
fig.suptitle(f'{MODEL}: is the recoverable spatial structure contained in the type-level baseline?',
             x=.01, ha='left', fontsize=11.5)
plt.tight_layout(rect=[0,0,1,.95]); plt.savefig(f'residualise_{tag}.png', dpi=190)
plt.show()

# ----
def _corr(e1, e2, yv, partial_y=True):
    # Two weak probes both predict near the mean, so both error vectors approach
    # -(y - mean) and correlate at ~.98 for reasons that have nothing to do with a
    # shared mechanism. Partialling y out of both removes exactly that component.
    if partial_y:
        D = np.column_stack([np.ones(len(yv)), yv])
        e1 = e1 - D @ np.linalg.lstsq(D, e1, rcond=None)[0]
        e2 = e2 - D @ np.linalg.lstsq(D, e2, rcond=None)[0]
    if np.std(e1) < 1e-12 or np.std(e2) < 1e-12: return np.nan
    return float(np.corrcoef(e1, e2)[0, 1])

def fold_error_corr(F1, F2, y, folds, restrict_within_continent=False, partial_y=True):
    # n-weighted mean correlation of the two probes' signed errors, computed inside
    # each test fold. Every city is covered; nothing is averaged across folds.
    acc = []
    for tr, te in folds:
        e1 = fit(F1[tr], y[tr]).predict(F1[te]) - y[te]
        e2 = fit(F2[tr], y[tr]).predict(F2[te]) - y[te]
        if restrict_within_continent:
            for k in range(len(conts)):
                m = cid[te] == k
                if m.sum() >= 5 and np.std(e1[m]) > 0 and np.std(e2[m]) > 0:
                    acc.append((m.sum(), _corr(e1[m], e2[m], y[te][m])))
        elif len(te) >= 5 and np.std(e1) > 0 and np.std(e2) > 0:
            acc.append((len(te), _corr(e1, e2, y[te])))
    acc = [(a, b) for a, b in acc if np.isfinite(b)]
    if not acc: return float('nan')
    w = np.array([a for a, _ in acc], float); v = np.array([b for _, b in acc], float)
    return float(np.sum(w * v) / np.sum(w))

err_rows = []
for tn, y in targets.items():
    Lb = int(df[(df.static == 'GloVe') & (df.target == tn)].sort_values('full').layer.iloc[-1])
    A  = acts[Lb].astype(np.float64)
    for sname, S in STATIC.items():
        r_all = fold_error_corr(A, S, y, rand_folds)
        r_wit = fold_error_corr(A, S, y, rand_folds, restrict_within_continent=True)
        # baseline: how correlated are two probes that share NO representation?
        r_null = fold_error_corr(A, rng.normal(size=S.shape), y, rand_folds)
        r_raw  = fold_error_corr(A, S, y, rand_folds, partial_y=False)
        err_rows.append(dict(target=tn, static=sname, layer=Lb, err_r=r_all,
                             err_r_within_continent=r_wit, err_r_random_features=r_null,
                             err_r_raw_uncorrected=r_raw))
        print(f'{tn:12s} LLM L{Lb:<3d} vs {sname:9s}  error r = {r_all:+.3f}   '
              f'within-continent {r_wit:+.3f}   (random-feature floor {r_null:+.3f}; '
              f'uncorrected {r_raw:+.3f})')
pd.DataFrame(err_rows).to_csv(f'error_correlation_{tag}.csv', index=False)

# ----
from scipy import stats

nb_rows = []
for tn, y in targets.items():
    Lb = int(df[(df.static == 'GloVe') & (df.target == tn)].sort_values('full').layer.iloc[-1])
    A  = acts[Lb].astype(np.float64)
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Sim = An @ An.T; np.fill_diagonal(Sim, -np.inf)
    recs = []
    for tr, te in rand_folds:
        p = fit(A[tr], y[tr]).predict(A[te])
        for j, c in enumerate(te):
            s = np.sort(Sim[c, tr])[::-1]
            recs.append((abs(p[j] - y[c]), s[0], s[:5].mean(), cid[c]))
    e   = np.array([r[0] for r in recs]); m1 = np.array([r[1] for r in recs])
    m5  = np.array([r[2] for r in recs]); gc = np.array([r[3] for r in recs])
    for nm, sim in [('max_sim', m1), ('top5', m5)]:
        r_raw = float(np.corrcoef(e, sim)[0, 1])
        # partial out continent with dummies
        D = np.column_stack([np.ones(len(e))] + [(gc == k).astype(float) for k in range(1, len(conts))])
        res_e   = e   - D @ np.linalg.lstsq(D, e,   rcond=None)[0]
        res_sim = sim - D @ np.linalg.lstsq(D, sim, rcond=None)[0]
        r_w = float(np.corrcoef(res_e, res_sim)[0, 1])
        t   = r_w * np.sqrt((len(e) - len(conts) - 1) / max(1e-12, 1 - r_w**2))
        p_w = float(2 * stats.t.sf(abs(t), len(e) - len(conts) - 1))
        nb_rows.append(dict(target=tn, layer=Lb, sim_measure=nm,
                            r_raw=r_raw, r_within_continent=r_w, p_within=p_w, n=len(e)))
        print(f'{tn:12s} L{Lb:<3d} {nm:8s}  r(|error|, sim) = {r_raw:+.3f}   '
              f'within-continent {r_w:+.3f}  (p={p_w:.2g})')
pd.DataFrame(nb_rows).to_csv(f'neighbour_dependence_{tag}.csv', index=False)

# ----
VOCAB = {"Cardinal directions": ["north", "south", "east", "west", "northern", "southern", "eastern", "western", "northeast", "northwest", "southeast", "southwest", "northward", "southward", "eastward", "westward"], "Climate & weather": ["cold", "warm", "hot", "cool", "freezing", "mild", "tropical", "arctic", "temperate", "humid", "arid", "dry", "monsoon", "snow", "rain", "sunny", "winter", "summer", "desert", "ice", "frost", "heat", "climate", "weather", "equatorial", "subtropical", "polar"], "Region & continent": ["europe", "european", "asia", "asian", "africa", "african", "america", "american", "oceania", "pacific", "atlantic", "mediterranean", "caribbean", "scandinavian", "nordic", "latin", "middle", "southeast", "western", "eastern", "central", "continental", "hemisphere", "equator", "arctic", "antarctic", "tropical", "subtropical"], "Country names": ["united", "states", "canada", "canadian", "mexico", "mexican", "britain", "british", "england", "english", "france", "french", "germany", "german", "spain", "spanish", "italy", "italian", "japan", "japanese", "china", "chinese", "india", "indian", "russia", "russian", "brazil", "brazilian", "australia", "australian", "egypt", "egyptian", "nigeria", "nigerian", "kenya", "kenyan", "turkey", "turkish", "iran", "iranian", "iraq", "iraqi", "korea", "korean", "thailand", "pakistan", "indonesian", "south", "zealand", "colombian", "argentina", "chilean", "peruvian", "venezuelan", "swedish", "norwegian", "finnish", "dutch", "belgian", "austrian", "czech", "polish", "hungarian", "portuguese", "irish", "scottish", "greek", "danish"], "Economic terms": ["rich", "poor", "wealthy", "poverty", "developed", "developing", "industrial", "economy", "economic", "gdp", "trade", "market", "financial", "business", "commercial", "investment", "growth", "income", "prosperity", "infrastructure", "modern", "urban", "metropolitan", "cosmopolitan", "capital", "port", "hub"], "Cultural & language": ["english", "french", "spanish", "chinese", "arabic", "hindi", "japanese", "german", "russian", "portuguese", "korean", "turkish", "persian", "dutch", "italian", "swedish", "muslim", "christian", "buddhist", "hindu", "catholic", "colonial", "imperial", "ancient", "historic", "medieval", "cultural", "heritage", "tradition", "cuisine", "language"]}

for k, v in VOCAB.items(): print(f'{k:22s} {len(v)} words')

# ----
from sklearn.decomposition import PCA

def word_acts(words, L):
    out = []
    with torch.no_grad():
        for wd in words:
            ids = tok(wd, return_tensors='pt').to(dev)
            hs  = model(**ids, output_hidden_states=True).hidden_states
            out.append(hs[L][0, -1].float().cpu().numpy())
    return np.array(out, dtype=np.float64)

abl_rows = []
for tn, y in targets.items():
    Lb = int(df[(df.static == 'GloVe') & (df.target == tn)].sort_values('full').layer.iloc[-1])
    A  = acts[Lb].astype(np.float64)
    base = probe(A, y, rand_folds)
    for cname, words in VOCAB.items():
        W = word_acts(words, Lb)
        k = min(len(words) - 1, 20)
        B = PCA(n_components=k).fit(W - W.mean(0)).components_
        Q = np.linalg.qr(B.T)[0]
        Ar = A - (A @ Q) @ Q.T
        r  = probe(Ar, y, rand_folds)
        abl_rows.append(dict(target=tn, layer=Lb, vocabulary=cname, k=k,
                             baseline=base, ablated=r, drop=base - r))
        print(f'{tn:12s} L{Lb:<3d} {cname:22s} k={k:2d}  {base:.3f} -> {r:+.3f}  '
              f'drop {base - r:+.3f}', flush=True)
pd.DataFrame(abl_rows).to_csv(f'llm_ablation_{tag}.csv', index=False)

# ----
import glob
try:
    for f in glob.glob(f'*{tag}*.csv') + glob.glob(f'*{tag}*.png'):
        print('downloading', f); files.download(f)
    print('\\nNOTE: acts_%s.npy is large; download it only if you want to redo the '
          'analyses without re-running extraction.' % tag)
except Exception as e:
    print('not in Colab; files are in the working directory:', e)