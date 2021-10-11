SELECT industry_id2,ad_bucket_num,flow_bucket_num,avg(y1),avg(y2),avg(y3) FROM data
GROUP BY industry_id2,ad_bucket_num,flow_bucket_num HAVING COUNT(*)>5
ORDER BY industry_id2;

SELECT industry_id2,COUNT(*) AS ad_num FROM data
GROUP BY industry_id2
ORDER BY ad_num DESC