FROM node:20-bookworm AS builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend ./
RUN npm run build

FROM node:20-bookworm-slim

WORKDIR /app/frontend

ENV NODE_ENV=production
ENV PORT=3000

COPY --from=builder /app/frontend ./

EXPOSE 3000

CMD ["npm", "run", "start"]
