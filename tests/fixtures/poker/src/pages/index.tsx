import { avatars } from "@/widgets/InputName";
import { Box, Button, Container, Stack, Typography } from "@mui/material";
import { sample } from "lodash";
import Link from "next/link";
import * as React from "react";

const adjectives1 = ["wild", "wacky", "wonderful", "wicked", "wavy", "wobbly"];
const adjectives2 = ["fuzzy", "furry", "fluffy", "fierce", "fancy", "fickle"];
const animals = avatars.map((avatar) => avatar.name);

export default function Home() {
  const [roomLink, setRoomLink] = React.useState<string | undefined>();
  React.useEffect(() => {
    const roomId = `${sample(adjectives1)}-${sample(adjectives2)}-${sample(
      animals
    )}`.toLocaleLowerCase();
    setRoomLink(`room/${roomId}`);
    for (const avatar of avatars) {
      new Image().src = avatar.avatar;
    }
  }, []);

  return (
    <Container maxWidth="lg">
      <Stack direction="row" alignItems="center" spacing={4} pt={8}>
        <Stack direction="column" spacing={2} alignItems="flex-start">
          <Typography variant="h1">
            <b>Wild Cards</b>
          </Typography>
          <Typography variant="subtitle1">
            {`The wildest planning poker app in on this planet.`}
          </Typography>
          <Stack direction="row" justifyContent="center" sx={{ width: "100%" }}>
            <Link href={roomLink ?? "/"}>
              <Button fullWidth variant="contained" size="large">
                {`Start Estimating`}
              </Button>
            </Link>
          </Stack>
        </Stack>
        <Box>
          <img
            src="/screenshot-in-browser-grey.png"
            alt="Wild Cards"
            style={{
              maxWidth: 650,
              borderRadius: 10,
            }}
          />
        </Box>
      </Stack>
    </Container>
  );
}
