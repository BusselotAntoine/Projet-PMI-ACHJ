%% --- PARAMÈTRES PHYSIQUES ET GÉOMÉTRIQUES ---
R = 0.01;       % Rayon du disque (1 cm)
D = 0.01;       % Distance entre les disques (1 cm)
epsilon = D/R;  % Paramètre sans dimension (rapport de forme)

a = 0; % Borne inférieure du domaine normalisé
b = 1; % Borne supérieure du domaine normalisé

% Noyau de l'équation intégrale de Love (modélise l'interaction électrostatique)
% K(x,y) est symétrique et dépend de la distance relative entre les points
K = @(x,y)(1/pi) * (1./(epsilon^2 + (x-y).^2) + 1./(epsilon^2 + (x+y).^2));

%% --- CONFIGURATION DES TESTS DE CONVERGENCE ---
n_values = [10, 20, 40, 80];        % Différentes discrétisations à tester
errors_classic = zeros(size(n_values)); % Stockage erreurs (méthode intégrale native)
errors_gauss   = zeros(size(n_values)); % Stockage erreurs (méthode quadrature Gauss)

%% --- RÉSOLUTIONS LINÉAIRES ---
% =========================================================================
% FONCTION : solve_linear_classic
% ROLE : Résout l'équation intégrale de Love par une méthode de Galerkin
%        avec des fonctions de base linéaires (P1) et utilise l'intégrateur
%        adaptatif natif de MATLAB pour remplir la matrice.
% -------------------------------------------------------------------------
% ENTRES :
%   n   : (Scalaire) Nombre d'intervalles de discrétisation.
%   K   : (Handle) Fonction du noyau K(x,y).
%   a,b : (Scalaire) Bornes du domaine d'intégration.
% SORTIES :
%   x   : (Vecteur) Points de la grille de discrétisation.
%   u   : (Vecteur) Solution (densité de charge) aux points x.
% =========================================================================
function [x, u] = solve_linear_classic(n, K, a, b)
    h = (b - a)/n;
    x = linspace(a, b, n+1);
    A = zeros(n+1);

    for i = 1:n+1
        for j = 1:n+1
            % Base de fonctions "chapeau" (P1 linéaire par morceaux)
            phi_j = @(y) max(0, 1 - abs(y - x(j))/h);
            y_min = max(a, x(j) - h);
            y_max = min(b, x(j) + h);
            % Calcul de l'élément de matrice par intégration adaptive
            A(i,j) = integral(@(y) K(x(i), y).*phi_j(y), y_min, y_max);
        end
    end
    % Résolution du système linéaire (I - A)u = 1
    u = (eye(n+1) - A) \ ones(n+1,1);
end

% =========================================================================
% FONCTION : gauss_legendre
% ROLE : Calcule les nœuds et les poids de la quadrature de Gauss-Legendre
%        sur l'intervalle standard [-1, 1] en utilisant l'algorithme de 
%        Golub-Welsch (diagonalisation de la matrice de Jacobi).
% -------------------------------------------------------------------------
% ENTRES :
%   n   : (Scalaire) Ordre de la quadrature (nombre de points).
% SORTIES :
%   x   : (Vecteur) Nœuds de quadrature sur [-1, 1].
%   w   : (Vecteur) Poids de quadrature associés.
% =========================================================================
function [x,w] = gauss_legendre(n)
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1); % Matrice de Jacobi
    [V,D] = eig(T);
    x = diag(D);
    [x, idx] = sort(x);
    w = 2 * (V(1,idx)').^2; % Poids de quadrature
end

% =========================================================================
% FONCTION : solve_linear_gauss
% ROLE : Similaire à solve_linear_classic, mais remplace l'intégrateur 
%        natif par une quadrature de Gauss-Legendre de degré fixe.
%        C'est beaucoup plus performant pour les calculs itératifs.
% -------------------------------------------------------------------------
% ENTRES :
%   n   : (Scalaire) Nombre d'intervalles.
%   K   : (Handle) Le noyau de l'équation.
%   a,b : (Scalaire) Bornes du domaine.
% SORTIES :
%   x   : (Vecteur) Grille de points.
%   u   : (Vecteur) Solution approchée.
% =========================================================================
function [x, u] = solve_linear_gauss(n, K, a, b)
    h = (b - a)/n;
    x = linspace(a, b, n+1);
    A = zeros(n+1);

    quad_order = 20; % Ordre de la quadrature pour chaque segment
    [gl_x, gl_w] = gauss_legendre(quad_order);

    for i = 1:n+1
        xi = x(i);
        for j = 1:n+1
            phi_j = @(y) max(0, 1 - abs(y - x(j))/h);
            y_min = max(a, x(j) - h);
            y_max = min(b, x(j) + h);

            if y_max <= y_min, A(i,j) = 0; continue; end

            % Changement de variable vers l'intervalle [y_min, y_max]
            yq = (y_max + y_min)/2 + (y_max - y_min)/2 * gl_x;
            wq = gl_w * (y_max - y_min)/2;

            A(i,j) = sum( K(xi, yq).*phi_j(yq).*wq );
        end
    end
    u = (eye(n+1) - A) \ ones(n+1,1);
end

%% --- CALCUL DE L'ERREUR ET RÉFÉRENCE ---

n_ref = 800; % Solution très fine pour servir de "vérité terrain"
[x_ref, u_ref] = solve_linear_gauss(n_ref, K, a, b);

for k = 1:length(n_values)
    n = n_values(k);
    
    % Test méthode classique
    [x_c, u_c] = solve_linear_classic(n, K, a, b);
    u_ref_interp = interp1(x_ref, u_ref, x_c);
    errors_classic(k) = max(abs(u_c - u_ref_interp'));

    % Test méthode Gauss
    [x_g, u_g] = solve_linear_gauss(n, K, a, b);
    u_ref_interp2 = interp1(x_ref, u_ref, x_g);
    errors_gauss(k) = max(abs(u_g - u_ref_interp2'));
end

%% --- APPROCHE PAR INTERPOLATION GLOBALE (LAGRANGE) ---

% =========================================================================
% FONCTION : lagrange_basis
% ROLE : Calcule la valeur du j-ième polynôme de base de Lagrange 
%        passant par les nœuds 'x_nodes' évalué au(x) point(s) 'y'.
% -------------------------------------------------------------------------
% ENTRES :
%   j       : (Entier) Indice du nœud de base (1 à n).
%   x_nodes : (Vecteur) Points d'interpolation (nœuds).
%   y       : (Vecteur/Scalaire) Point(s) où évaluer le polynôme.
% SORTIES :
%   Lj      : (Vecteur/Scalaire) Valeur(s) de phi_j(y).
% =========================================================================
function Lj = lagrange_basis(j, x_nodes, y)
    n = length(x_nodes);
    Lj = ones(size(y));
    xj = x_nodes(j);
    for k = 1:n
        if k ~= j
            Lj = Lj .* ( (y - x_nodes(k)) / (xj - x_nodes(k)) );
        end
    end
end

% =========================================================================
% FONCTION : build_A_matrix
% ROLE : Construit la matrice de l'opérateur intégral en utilisant une
%        approximation par interpolation globale (Lagrange) au lieu 
%        d'une approximation locale par morceaux.
% -------------------------------------------------------------------------
% ENTRES :
%   x_nodes    : (Vecteur) Points d'interpolation choisis.
%   kernel     : (Handle) Fonction noyau K(x,y).
%   a,b        : (Scalaire) Intervalle d'intégration.
%   quad_order : (Entier) Nombre de points pour la quadrature numérique.
% SORTIES :
%   A          : (Matrice n x n) Matrice de Fredholm discrétisée.
% =========================================================================
function A = build_A_matrix(x_nodes, kernel, a, b, quad_order)
    n = length(x_nodes);
    A = zeros(n);
    [yq, wq] = gauss_legendre(quad_order);
    yq = 0.5*(b-a)*yq + 0.5*(b+a);
    wq = 0.5*(b-a)*wq;

    for i = 1:n
        xi = x_nodes(i);
        for j = 1:n
            Lj = lagrange_basis(j, x_nodes, yq);
            A(i,j) = sum((kernel(xi, yq) .* Lj) .* wq);
        end
    end
end

%% --- ANALYSE DES RÉSULTATS ET AFFICHAGE ---

% Question 5
fprintf('Q5\n');

n2 = 40; % Nombre de points
x2 = linspace(a, b, n2); % Points
A2 = build_A_matrix(x2,K,a,b,n2);
I2 = eye(n2);
d2 = ones(n2, 1); % Terme source d(x) = 1

u2 = (I2 - A2) \ d2;

figure(1);
plot(x_c, u_c);
title('Densité de charge équivalente u(x)');
xlabel('Position normalisée x');
ylabel('u(x)');


figure(2);
plot(x_g, u_g, 'LineWidth', 2);
title('Méthode linéaire avec quadrature de Gauss-Legendre');
xlabel('x');
ylabel('u(x)');
grid on;

figure(3);
plot(x2, u2);
title('Densité de charge équivalente u2(x)');
xlabel('Position normalisée x2');
ylabel('u2(x)');

fprintf('Regarder figure 1,2 & 3\n');

fprintf('\n');
% Question 7
fprintf('Q7\n');


% Graphique Erreur vs n 

figure(4);
subplot(1,2,1);
loglog(n_values, errors, 'r-o', 'LineWidth', 2);
grid on;
xlabel('Nombre de points n');
ylabel('Erreur e_n');
title('Erreur vs n');

% Graphique Erreur vs h 
subplot(1,2,2);
loglog(h_values, errors, 'b-s', 'LineWidth', 2);
title('Erreur vs h');
xlabel('Pas d''espace h');
ylabel('Erreur e_n');
hold on;
grid on;

% --- Ajout : droite de référence en h^2 ---
ref = errors(1) * (h_values / h_values(1)).^2;
loglog(h_values, ref, 'r--', 'LineWidth', 1.5);
legend('Erreur', 'Référence h^2');

% --- Ajout : calcul de l'ordre ---
p = polyfit(log(h_values), log(errors), 1);
disp(['Ordre de convergence estimé : p = ', num2str(p(1))]);

% Question 8 ----------
fprintf('Q8\n');

figure(5);
loglog(n_values, errors_classic, 'r-o', 'LineWidth', 2); hold on;
loglog(n_values, errors_gauss, 'b-s', 'LineWidth', 2);
grid on;
xlabel('n');
ylabel('Erreur max ||u - u_n||_\infty');
title('Comparaison des méthodes linéaires');
legend('Classique (integral)', 'Gauss-Legendre');
fprintf('Regarder figure 4 & 5\n');


fprintf('\n');

% Question 9 ----------
fprintf('Q9\n');
epsilon0 = 8.854e-12; 
R_val = 0.01;         
D_val = 0.01;         
epsilon_geom = D_val / R_val; 

K1 = @(x,y)(1/pi) * (1./(epsilon_geom^2 + (x-y).^2) + 1./(epsilon_geom^2 + (x+y).^2));

n_ref = 200; 
[x_res, u_res] = solve_linear_gauss(n_ref, K1, 0, 1);

valeur_integrale = trapz(x_res, u_res); 
C_numerique = 4 * epsilon0 * R_val * valeur_integrale;

fprintf('Résultat de l''intégrale numérique : %.4f\n', valeur_integrale);
fprintf('Capacité calculée (Love) : %.4e F\n', C_numerique);

fprintf('\n');

% Q10
fprintf('Q10\n');
     
S = pi * R_val^2;     

C_classique = (epsilon0 * S) / D_val;

fprintf('Capacité classique (S/D) : %.4e F\n', C_classique);
ecart_relatif = abs(C_numerique - C_classique) / C_numerique * 100;
fprintf('Écart relatif : %.2f %%\n', ecart_relatif);

D_values = [0.01 0.001]; 

fprintf('\n');

% Q11
fprintf('Q11\n');
fprintf('Regarder figure 6\n');
for D = D_values
    epsilon = D/R_val;
    K_new = @(x,y)(1/pi) * (1./(epsilon^2 + (x-y).^2) + 1./(epsilon^2 + (x+y).^2));
    
    % Résolution (utiliser la fonction solve_linear_gauss de votre code)
    n_pts = 100;
    [x_u, u_sol] = solve_linear_gauss(n_pts, K_new, 0, 1);
    
    % Calculs des capacités
    C_Love = 4 * epsilon0 * R * trapz(x_u, u_sol);
    C_class = (epsilon0 * S) / D;
    

    % Tracer u(x) pour comparer les profils
    figure(6); hold on;
    plot(x_u, u_sol, 'DisplayName', ['D = ' num2str(D*1000) ' mm']);
end

title('Influence de la distance sur la densité de charge u(x)');
xlabel('x (normalisé)'); ylabel('u(x)');
legend; grid on; 

fprintf('\n');
