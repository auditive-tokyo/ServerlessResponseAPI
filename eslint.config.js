import tseslint from 'typescript-eslint';
import prettierConfig from 'eslint-config-prettier';

export default tseslint.config(
  // グローバルに無視するファイル/ディレクトリを指定
  {
    ignores: ['dist/', 'node_modules/', '.aws-sam/'],
  },
  // typescript-eslint の推奨設定を適用
  ...tseslint.configs.recommended,
  // カスタムルール設定
  {
    rules: {
      // アンダースコアプレフィックスの未使用変数を許可
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
    },
  },
  // Prettier の設定を適用し、競合するルールを無効化
  prettierConfig
);