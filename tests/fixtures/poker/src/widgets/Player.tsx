import {
  Avatar,
  Badge,
  Stack,
  Typography
  } from "@mui/material";
import { PlayingCard } from "./PlayingCard";

export type Player = {
  name: string;
  avatar: string;
  estimate?: number | null;
  isMe?: boolean;
  isLeader?: boolean;
  reveal?: boolean;
  sequence?: boolean;
};

export const PlayerPresenter: React.FC<Player> = ({
  name,
  avatar,
  estimate,
  reveal = false,
  isLeader = false,
  isMe = false,
}) => {
  return (
    <Stack
      direction="column"
      justifyContent="center"
      alignItems="center"
      sx={{
        width: { xs: 150, md: 200 },
        borderWidth: 4,
        borderColor: isMe ? "secondary.main" : "primary.main",
        borderStyle: "solid",
        borderRadius: 2,
        p: 2,
      }}
    >
      <Badge badgeContent={isLeader ? "⭐️" : ""} color="default">
        <Avatar
          alt={name}
          src={avatar}
          sx={{
            height: 100,
            width: 100,
          }}
        />
      </Badge>
      <Typography textAlign="center" variant="h6">
        {name}
      </Typography>
      <PlayingCard isFlipped={reveal} text={estimate} />
    </Stack>
  );
};

export default PlayerPresenter;
