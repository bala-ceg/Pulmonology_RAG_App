require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const { Pool } = require('pg');

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(__dirname));

const pool = new Pool({
  host: process.env.PG_TOOL_HOST || process.env.DB_HOST,
  port: Number(process.env.PG_TOOL_PORT || process.env.DB_PORT || 5432),
  database: process.env.PG_TOOL_NAME || process.env.DB_NAME,
  user: process.env.PG_TOOL_USER || process.env.DB_USER,
  password: process.env.PG_TOOL_PASSWORD || process.env.DB_PASSWORD,
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

app.post('/api/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    if (!username || !password) {
      return res.status(400).json({ success: false, message: 'Username and password are required' });
    }

    const columnResult = await pool.query(
      "SELECT LOWER(column_name) AS column_name FROM information_schema.columns WHERE table_name = 'p_party'"
    );
    const columns = new Set(columnResult.rows.map(row => row.column_name));
    const loginColumns = ['username', 'user_name', 'email'].filter(column => columns.has(column));
    const passwordColumn = ['password_hash', 'password'].find(column => columns.has(column));
    const roleColumn = ['pces_role', 'role', 'specialty', 'department'].find(column => columns.has(column));

    if (!loginColumns.length || !passwordColumn) {
      console.error('p_party login schema missing required columns', { loginColumns, passwordColumn });
      return res.status(500).json({ success: false, message: 'Login is not configured for the EHR schema' });
    }

    const usernameColumn = columns.has('username') ? 'username' : loginColumns[0];
    const predicates = loginColumns.map((column, index) => `LOWER(${column}) = LOWER($${index + 1})`).join(' OR ');
    const queryText = `
      SELECT
        party_id,
        ${usernameColumn} AS username,
        ${passwordColumn} AS password_hash,
        ${roleColumn || 'NULL'} AS pces_role
      FROM p_party
      WHERE party_type = 'DOCTOR'
        AND COALESCE(is_active, TRUE) = TRUE
        AND (${predicates})
      LIMIT 1
    `;
    const { rows } = await pool.query(queryText, loginColumns.map(() => username));

    if (!rows.length) {
      return res.status(401).json({ success: false, message: 'Invalid username or password' });
    }

    const user = rows[0];
    const match = password === user.password_hash;

    if (!match) {
      return res.status(401).json({ success: false, message: 'Invalid username or password' });
    }

    return res.json({
      success: true,
      username: user.username,
      party_id: user.party_id,
      pces_role: user.pces_role,
      message: `Welcome ${user.username}, your role is ${user.pces_role}`,
    });
  } catch (error) {
    console.error('Login error:', error);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
});

app.get('/health', (req, res) => {
  return res.json({ status: 'ok' });
});

app.listen(port, () => {
  console.log(`Login server running at http://localhost:${port}`);
});
