import {
  Avatar,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import * as React from "react";
import { Player } from "./Player";

export const avatars = [
  { name: "Bighorn Sheep", avatar: "/avatars/bighorn-sheep.png" },
  { name: "Crocodile", avatar: "/avatars/crocodile.png" },
  { name: "Elephant", avatar: "/avatars/elephant.png" },
  { name: "Gorilla", avatar: "/avatars/gorilla.png" },
  { name: "Kangaroo", avatar: "/avatars/kangaroo.png" },
  { name: "Leopard", avatar: "/avatars/leopard.png" },
  { name: "Otter", avatar: "/avatars/otter.png" },
  { name: "Peacock", avatar: "/avatars/peacock.png" },
  { name: "Pig", avatar: "/avatars/pig.png" },
  { name: "Red Panda", avatar: "/avatars/redpanda.png" },
  { name: "Sloth", avatar: "/avatars/sloth.png" },
  { name: "Zebra", avatar: "/avatars/zebra.png" },
  { name: "Trout", avatar: "/avatars/trout.png" },
  { name: "Bison", avatar: "/avatars/bison.png" },
  { name: "Penguin", avatar: "/avatars/penguin.png" },
];

type Props = {
  onComplete: (player: Player) => void;
};

export const InputName: React.FC<Props> = ({ onComplete }) => {
  const [name, setName] = React.useState<string>();
  const [selectedAvatar, setSelectedAvatar] = React.useState<string>("");
  return (
    <Dialog open={true} maxWidth="sm" fullWidth>
      <DialogContent>
        <Stack direction="column" spacing={2}>
          <Typography variant="h4" textAlign="center">
            {`What's your name?`}
          </Typography>
          <TextField
            fullWidth
            label="Your Name"
            variant="standard"
            size="medium"
            onChange={(e) => setName(e.target.value)}
            value={name ?? ""}
          />
          <Typography variant="h4" textAlign="center">
            Select an Avatar
          </Typography>
          <Stack
            direction="row"
            justifyContent="center"
            spacing={2}
            useFlexGap
            flexWrap="wrap"
          >
            {avatars.map((avatar) => (
              <Avatar
                key={`avatar-${avatar.name}`}
                alt={avatar.name}
                src={avatar.avatar}
                onClick={() => setSelectedAvatar(avatar.avatar)}
                sx={{
                  height: { xs: "60%", md: 96 },
                  width: { xs: "60%", md: 96 },
                  borderWidth: avatar.avatar === selectedAvatar ? 4 : 0,
                  borderColor: "primary.main",
                  borderStyle: "solid",
                  "&:hover": {
                    cursor: "pointer",
                  },
                }}
              />
            ))}
          </Stack>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Stack direction="row" justifyContent="space-between" width={"100%"}>
          <Button
            onClick={() => {
              onComplete({
                name: "Guest",
                avatar: "/avatars/guest.png",
              });
            }}
          >
            Join As Guest
          </Button>
          <Button
            variant="contained"
            color="primary"
            disabled={!name || !selectedAvatar}
            onClick={() =>
              onComplete({
                name: name ?? "",
                avatar: selectedAvatar,
              })
            }
          >
            {`Let's go!`}
          </Button>
        </Stack>
      </DialogActions>
    </Dialog>
  );
};

export default InputName;
