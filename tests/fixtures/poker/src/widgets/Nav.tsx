import Link from "next/link";
import { Stack, Typography } from "@mui/material";

export const Nav: React.FC = () => (
  <Stack
    direction="row"
    bgcolor="primary.main"
    justifyContent="space-between"
    sx={{ p: 1 }}
  >
    <Link href="/" style={{ textDecoration: "none" }}>
      <Stack direction="row" spacing={1}>
        <img src={"/logo.png"} alt="Red Panda" height={32} width={32} />
        <Typography
          variant="h6"
          color="white"
          sx={{
            textDecoration: "none",
          }}
        >
          Wild Cards
        </Typography>
      </Stack>
    </Link>
  </Stack>
);

export default Nav;
