# Descomposición de varianza del Dice por imagen de VALIDACIÓN con un modelo lineal mixto (lme4).
#
#   "C:\Program Files\R\R-4.5.0\bin\Rscript.exe" scripts/varianza_mixta_val_e2.R [B]
#
# Se corre desde la raíz del proyecto (tesis_imo_para_codex). B = réplicas del bootstrap por grupos (200 por
# omisión; 0 lo omite). Datos: resultados/validacion_por_imagen_e2/*_val_por_imagen.json (50 corridas = 10 redes x 5
# semillas, 218 imágenes de 122 grupos de similitud: 10.900 filas). Es VALIDACIÓN: el mismo conjunto con el que se
# eligió el checkpoint de cada corrida, así que el nivel de cada red está sesgado al alza; la descomposición describe
# de dónde viene la variación, no el desempeño esperado.
#
# Modelo principal (REML):
#   dice ~ red + (1|semilla) + (1|red:semilla) + (1|grupo) + (1|grupo:imagen) + (1|red:grupo) + residuo
#   - semilla: efecto común de la semilla a todas las redes (mismo orden de datos y aumentos).
#   - red:semilla: la corrida, lo propio de cada entrenamiento.
#   - grupo, grupo:imagen: dificultad de la escena y de la imagen dentro del grupo.
#   - red:grupo: interacción red x escena (unas redes aciertan donde otras fallan).
#   - residuo: semilla x imagen dentro de la red.
# El término red:grupo:imagen se probó y quedó en el borde (varianza 0): con dos imágenes por grupo como máximo, casi
# duplicadas, la interacción red x imagen la absorbe red:grupo. Se informa y se quita del modelo principal.
#
# El Dice está acotado en [0, 1] y tiene 96 ceros: los residuos no son normales. Alternativas robustas que se corren:
#   (a) el mismo modelo sobre la transformación normal inversa de rangos (INT) del Dice;
#   (b) el mismo modelo sin los Dice = 0 (fallos totales);
#   (c) IC95 percentil de las proporciones por bootstrap de grupos completos (los grupos se remuestrean con todas
#       sus imágenes, redes y semillas; cada réplica recibe un identificador nuevo de grupo).
# Salidas: resultados/inferencia_cierre_e2/varianza_mixta.json y varianza_mixta_diagnostico.png.

suppressMessages({library(lme4); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
B <- if (length(args) >= 1) as.integer(args[1]) else 200L
OUT <- "resultados/inferencia_cierre_e2"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
set.seed(20261122)

fs <- list.files("resultados/validacion_por_imagen_e2", pattern = "_val_por_imagen.json$", full.names = TRUE)
stopifnot(length(fs) == 50)
df <- do.call(rbind, lapply(fs, function(f) {
  j <- fromJSON(f); d <- j$detail
  stopifnot(j$split == "val")
  data.frame(red = j$model, semilla = as.character(j$seed), grupo = d$group_id, imagen = d$id, dice = d$Dice)
}))
stopifnot(nrow(df) == 10900, !anyNA(df$dice))
df$red <- relevel(factor(df$red), ref = "fcn")
df$int <- qnorm((rank(df$dice, ties.method = "average") - 0.5) / nrow(df))

formula_principal <- . ~ red + (1 | semilla) + (1 | red:semilla) + (1 | grupo) + (1 | grupo:imagen) + (1 | red:grupo)
ajustar <- function(datos, resp = "dice") {
  f <- update(as.formula(paste(resp, "~ 1")), formula_principal)
  suppressMessages(lmer(f, data = datos, REML = TRUE,
                        control = lmerControl(calc.derivs = FALSE)))
}
componentes <- function(m) {
  vc <- as.data.frame(VarCorr(m))
  v <- setNames(vc$vcov, vc$grp)
  fijo <- var(as.vector(model.matrix(m) %*% fixef(m)))   # varianza de la parte fija (Nakagawa)
  aleat <- v[c("semilla", "red:semilla", "grupo", "grupo:imagen", "red:grupo", "Residual")]
  list(varianza = as.list(aleat), varianza_red_fija = fijo,
       prop_aleatoria = as.list(aleat / sum(aleat)),
       prop_total = as.list(c(red_fija = fijo, aleat) / (fijo + sum(aleat))))
}

# Término a tres vías: se ajusta para documentar que queda en el borde.
m3 <- suppressMessages(lmer(dice ~ red + (1 | semilla) + (1 | red:semilla) + (1 | grupo) + (1 | grupo:imagen) +
                              (1 | red:grupo) + (1 | red:grupo:imagen), data = df, REML = TRUE))
v3 <- as.data.frame(VarCorr(m3)); v3 <- setNames(v3$vcov, v3$grp)

m <- ajustar(df)
comp <- componentes(m)

# Pruebas de razón de verosimilitud (REML, misma parte fija) para los componentes de semilla; la nula está en el
# borde del espacio paramétrico, así que p se divide entre 2 (mezcla 50:50 de chi2_0 y chi2_1).
lrt <- function(sin) {
  f <- update(as.formula("dice ~ 1"), formula_principal)
  f0 <- update(f, as.formula(paste(". ~ . -", sin)))
  m0 <- suppressMessages(lmer(f0, data = df, REML = TRUE))
  stat <- as.numeric(2 * (logLik(m) - logLik(m0)))
  list(termino = sin, estadistico = stat, p_mezcla = 0.5 * pchisq(stat, 1, lower.tail = FALSE))
}
pruebas <- list(lrt("(1 | semilla)"), lrt("(1 | red:semilla)"), lrt("(1 | red:grupo)"))

# Efecto fijo de red: F con Satterthwaite si lmerTest está disponible.
f_red <- tryCatch({
  suppressMessages(library(lmerTest))
  mt <- as_lmerModLmerTest(m); a <- anova(mt)
  list(F = a$`F value`[1], gl_num = a$NumDF[1], gl_den = a$DenDF[1], p = a$`Pr(>F)`[1])
}, error = function(e) list(error = conditionMessage(e)))

# Diagnóstico de residuos del modelo principal.
r <- residuals(m); ft <- fitted(m)
asim <- function(x) mean((x - mean(x))^3) / sd(x)^3
curt <- function(x) mean((x - mean(x))^4) / sd(x)^4 - 3
blup_g <- ranef(m)$grupo[, 1]; blup_rg <- ranef(m)$`red:grupo`[, 1]
diag <- list(
  residuos_asimetria = asim(r), residuos_curtosis_exceso = curt(r),
  residuos_fuera_3DE = mean(abs(r) > 3 * sd(r)),
  corr_absres_ajustado = cor(abs(r), ft, method = "spearman"),
  ajustados_fuera_01 = mean(ft < 0 | ft > 1),
  blup_grupo_asimetria = asim(blup_g), blup_red_grupo_asimetria = asim(blup_rg),
  dice_cero = sum(df$dice == 0), singular_principal = isSingular(m))

png(file.path(OUT, "varianza_mixta_diagnostico.png"), width = 1500, height = 1000, res = 130)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))
plot(ft, r, pch = ".", col = rgb(0, 0, 0, .3), xlab = "Ajustado", ylab = "Residuo", main = "Residuo vs ajustado")
abline(h = 0, col = 2)
qqnorm(r, pch = ".", main = "QQ de residuos"); qqline(r, col = 2)
hist(r, breaks = 80, main = "Histograma de residuos", xlab = "Residuo")
qqnorm(blup_g, main = "QQ BLUP de grupo"); qqline(blup_g, col = 2)
qqnorm(blup_rg, pch = 20, cex = .4, main = "QQ BLUP red x grupo"); qqline(blup_rg, col = 2)
ri <- residuals(ajustar(df, "int"))
qqnorm(ri, pch = ".", main = "QQ residuos, modelo sobre rangos (INT)"); qqline(ri, col = 2)
dev.off()

# Alternativas robustas.
m_int <- ajustar(df, "int")
m_sin0 <- ajustar(df[df$dice > 0, ])
robustas <- list(int_rangos = componentes(m_int), sin_ceros = componentes(m_sin0))

# Bootstrap por grupos completos.
boot <- NULL
if (B > 0) {
  grupos <- unique(df$grupo); por_grupo <- split(df, df$grupo)
  nombres <- c("red_fija", "semilla", "red:semilla", "grupo", "grupo:imagen", "red:grupo", "Residual")
  boot <- matrix(NA_real_, B, length(nombres), dimnames = list(NULL, nombres))
  for (b in seq_len(B)) {
    sel <- sample(grupos, length(grupos), replace = TRUE)
    db <- do.call(rbind, lapply(seq_along(sel), function(k) {
      x <- por_grupo[[sel[k]]]; x$grupo <- paste0("b", k); x$imagen <- paste0("b", k, "_", x$imagen); x }))
    cb <- tryCatch(componentes(ajustar(db)), error = function(e) NULL)
    if (!is.null(cb)) boot[b, ] <- unlist(cb$prop_total)[nombres]
    if (b %% 20 == 0) cat("bootstrap", b, "de", B, "\n")
  }
  boot <- boot[complete.cases(boot), , drop = FALSE]
}
ic <- if (is.null(boot)) NULL else
  lapply(as.data.frame(boot), function(x) unname(quantile(x, c(.025, .975))))

medias_red <- aggregate(dice ~ red, df, mean)
res <- list(
  alcance = paste("VALIDACIÓN (selección de checkpoint): descriptivo de la estructura de varianza; exploratorio,",
                  "posterior al test. Unidad de observación: imagen x corrida; 10.900 filas."),
  n = list(filas = nrow(df), redes = nlevels(df$red), semillas = length(unique(df$semilla)),
           grupos = length(unique(df$grupo)), imagenes = length(unique(df$imagen))),
  formula = "dice ~ red + (1|semilla) + (1|red:semilla) + (1|grupo) + (1|grupo:imagen) + (1|red:grupo)",
  principal = comp,
  termino_tres_vias_red_grupo_imagen = list(varianza = unname(v3["red:grupo:imagen"]), singular = isSingular(m3)),
  lrt_componentes = pruebas, F_red = f_red, diagnostico = diag, robustas = robustas,
  bootstrap_grupos = list(B_validas = if (is.null(boot)) 0 else nrow(boot), IC95_prop_total = ic),
  efectos_fijos_vs_fcn = as.list(fixef(m)),
  dice_medio_por_red_imagenes = setNames(as.list(medias_red$dice), as.character(medias_red$red)))
write_json(res, file.path(OUT, "varianza_mixta.json"), auto_unbox = TRUE, pretty = TRUE, digits = 8)
cat("Proporción del total (principal):\n"); print(round(unlist(comp$prop_total), 4))
cat("Listo:", file.path(OUT, "varianza_mixta.json"), "\n")
