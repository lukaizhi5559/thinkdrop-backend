import * as winston from 'winston';

const isDevelopment = process.env.NODE_ENV !== 'production';
const logLevel = process.env.LOG_LEVEL || (isDevelopment ? 'debug' : 'info');

const winstonLogger = winston.createLogger({
  level: logLevel,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json(),
  ),
  defaultMeta: { service: 'thinkdrop-backend' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf((info) => {
          const { timestamp, level, message, ...meta } = info;
          return `${timestamp} [${level}]: ${message} ${
            Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ''
          }`;
        })
      ),
    }),
    ...(isDevelopment
      ? []
      : [
          new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
          new winston.transports.File({ filename: 'logs/combined.log' }),
        ]),
  ],
});

export const logger = {
  info: (message: string, meta: Record<string, any> = {}) => winstonLogger.info(message, meta),
  error: (message: string, meta: Record<string, any> = {}) => winstonLogger.error(message, meta),
  warn: (message: string, meta: Record<string, any> = {}) => winstonLogger.warn(message, meta),
  debug: (message: string, meta: Record<string, any> = {}) => winstonLogger.debug(message, meta),
};
