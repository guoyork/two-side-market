


with tb4 as
(SELECT simulation_id,req_id,adgroup_id,ecpm,rankno
FROM aaa
WHERE (simulation_id=1221 OR simulation_id=1231)
ORDER BY req_id)

/*
SELECT *
FROM tb4
ORDER BY ecpm1+0 DESC;
*/
SELECT simulation_id,SUM(ecpm),STDDEV(ecpm)
FROM tb4
GROUP BY simulation_id;


with cs as
(SELECT req_id,COUNT(DISTINCT adgroup_id) as counts
FROM aaa
WHERE (simulation_id=1221 OR simulation_id=1231)
GROUP BY req_id
ORDER BY req_id)

SELECT COUNT(*)
FROM cs
WHERE counts=1;




with tb3 as
(SELECT adgroup_id as id1,COUNT(DISTINCT req_id) as imp_count
FROM aaa
WHERE (simulation_id=1221 OR simulation_id=1231)
GROUP BY adgroup_id
ORDER BY adgroup_id)


SELECT imp_count, COUNT(id1) as freq
FROM tb3
GROUP BY imp_count
ORDER BY imp_count;



with tb1 as 
(SELECT a.req_id,a.adgroup_id as id1, b.adgroup_id as id2 FROM aaa a CROSS JOIN aaa b on a.req_id=b.req_id
WHERE (a.simulation_id=1221 OR a.simulation_id=1231) AND (b.simulation_id=1221 OR b.simulation_id=1231)
ORDER BY a.req_id),

tb2 as
(SELECT id1, COUNT(DISTINCT id2) as pair_count
FROM tb1
WHERE id1<>id2
GROUP BY id1)

SELECT pair_count, COUNT(id1) as freq
FROM tb2
GROUP BY pair_count
ORDER BY pair_count;


with tb5 as
(SELECT id1,id2,COUNT(*) as pair_count
FROM tb1
WHERE id1<>id2
GROUP BY id1,id2)

SELECT pair_count, COUNT(*) as freq
FROM tb5
GROUP BY pair_count
ORDER BY pair_count;
