UPDATE pg_settings SET setting = 1 WHERE name = 'join_collapse_limit';
UPDATE pg_settings SET setting = 1 WHERE name = 'max_parallel_workers';
UPDATE pg_settings SET setting = 1 WHERE name = 'max_parallel_workers_per_gather';
UPDATE pg_settings SET setting = 20 WHERE name = 'geqo_threshold';

EXPLAIN select min(chn.name) AS voiced_char,
min(n.name) AS voicing_actress,
min(t.title) AS voiced_animation
from cast_info AS ci
inner join role_type AS rt
on rt.id = ci.role_id AND ci.note IN ('(voice)',
'(voice: Japanese version)',
'(voice) (uncredited)',
'(voice: English version)') AND rt.role = 'actress'
inner join aka_name AS an
on ci.person_id = an.person_id
inner join complete_cast AS cc
on ci.movie_id = cc.movie_id
inner join movie_companies AS mc
on mc.movie_id = ci.movie_id AND mc.movie_id = cc.movie_id
inner join char_name AS chn
on chn.id = ci.person_role_id
inner join person_info AS pi
on ci.person_id = pi.person_id
inner join comp_cast_type AS cct1
on cct1.id = cc.subject_id AND cct1.kind = 'cast'
inner join movie_info AS mi
on mi.movie_id = ci.movie_id AND mi.movie_id = cc.movie_id AND mc.movie_id = mi.movie_id AND mi.info IS NOT NULL AND ( mi.info like 'Japan:%200%' OR mi.info like 'USA:%200%')
inner join info_type AS it
on it.id = mi.info_type_id AND it.info = 'release dates'
inner join info_type AS it3
on it3.id = pi.info_type_id AND it3.info = 'trivia'
inner join company_name AS cn
on cn.id = mc.company_id AND cn.country_code = '[us]'
inner join movie_keyword AS mk
on ci.movie_id = mk.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND mc.movie_id = mk.movie_id
inner join title AS t
on t.id = mi.movie_id AND t.id = cc.movie_id AND t.id = mk.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND t.production_year BETWEEN 2000 AND 2010
inner join name AS n
on n.id = ci.person_id AND n.id = pi.person_id AND n.id = an.person_id AND n.gender = 'f' AND n.name like '%An%'
inner join comp_cast_type AS cct2
on cct2.id = cc.status_id AND cct2.kind = 'complete+verified'
inner join keyword AS k
on k.id = mk.keyword_id AND k.keyword = 'computer-animation';