-- Start clean (optional)
DROP TABLE IF EXISTS calendar_dates;

CREATE TABLE calendar_dates (
    `date` DATE PRIMARY KEY
);

SET @start := DATE('2010-01-01');
SET @end   := DATE('2025-12-31');
SET @i := -1;

INSERT INTO calendar_dates (`date`)
SELECT d
FROM (
    SELECT DATE_ADD(@start, INTERVAL @i := @i + 1 DAY) AS d
    FROM information_schema.COLUMNS c1
    CROSS JOIN information_schema.COLUMNS c2
    -- c1 x c2 gives you *lots* of rows (hundreds of thousands),
    -- enough to cover all days
) AS dates
WHERE d <= @end;