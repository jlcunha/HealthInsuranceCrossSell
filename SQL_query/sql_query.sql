SELECT *
FROM pa004.users u INNER JOIN pa004.vehicle v   ON (u.id = v.id)
                   INNER JOIN pa004.insurance i ON (u.id = i.id)